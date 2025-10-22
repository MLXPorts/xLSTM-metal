#!/usr/bin/env python
#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""Metal kernel for parallel part of forward pass of the mLSTM chunkwise formulation.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes outputs H within each chunk in parallel using:
1. Intra-chunk: attention within chunk using causal mask
2. Inter-chunk: contribution from previous state C_{k-1}
3. Combine: H = (H_inter + ratio * H_intra) / denom
"""

import mlx.core as mx
from typing import Tuple

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_PARALLEL_FW_HINTRA_SRC = r"""
    // Thread and threadgroup indices
    uint idx_b_DHHV = threadgroup_position_in_grid.x;
    uint idx_b_LQ = threadgroup_position_in_grid.y;
    uint idx_b_NC_BNH = threadgroup_position_in_grid.z;

    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    // Extract dimensions from params buffer
    uint B = params[0];
    uint NH = params[1];
    uint S = params[2];
    uint DHQK = params[3];
    uint DHHV = params[4];
    uint NC = params[5];
    uint L = params[6];
    uint siz_b_LQ = params[7];
    uint siz_b_LKV = params[8];
    uint siz_b_DHQK = params[9];
    uint siz_b_DHHV = params[10];
    float qk_scale = as_type<float>(params[11]);  // Reinterpret as float
    float EPS = as_type<float>(params[12]);
    float MINIMUM_MAX_VAL = as_type<float>(params[13]);

    uint idx_b_NC = idx_b_NC_BNH % NC;
    uint idx_b_BNH = idx_b_NC_BNH / NC;

    // Extract strides from strides buffer
    uint str_matQK_B_NH = strides[0];
    uint str_matQK_S = strides[1];
    uint str_matQK_DHQK = strides[2];
    uint str_matHV_B_NH = strides[3];
    uint str_matHV_S = strides[4];
    uint str_matHV_DHHV = strides[5];
    uint str_matCstates_B_NH = strides[6];
    uint str_matCstates_NCDHQK = strides[7];
    uint str_matCstates_DHHV = strides[8];
    uint str_vecNstates_B_NH = strides[9];
    uint str_vecNstates_NCDHQK = strides[10];
    uint str_scaMinterstates_B_NH = strides[11];
    uint str_vecBI_B_NH = strides[12];
    uint str_vecBI_NC = strides[13];
    uint str_vecBI_L = strides[14];
    uint str_vecMN_B_NH = strides[15];
    uint str_vecMN_S = strides[16];

    // Initialize vecM states (thread-local) - sized for max tile
    thread float vecM_old_val[16];
    thread float vecM_new_val[16];
    for (uint i = 0; i < 16; ++i) {
        vecM_old_val[i] = -INFINITY;
        vecM_new_val[i] = -INFINITY;
    }

    // Load vecB_LQ and compute gate pointers
    uint vecB_base = idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC;
    uint vecI_base = idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC;

    thread float vecB_LQ_val[16];
    for (uint i = tx; i < siz_b_LQ && i < 16; i += siz_b_DHHV) {
        uint idx = vecB_base + idx_b_LQ * siz_b_LQ + i;
        vecB_LQ_val[i] = vecB[idx];
    }

    // For causal masking
    uint b_q_offset = idx_b_LQ * siz_b_LQ;

    // Threadgroup memory - sized for 16x16 tiles (within 32KB limit)
    // Max usage: ~7 arrays * 16x16 * 4 bytes = ~7KB
    threadgroup float matH_intra_acc[16][16];
    threadgroup float vecN_intra_acc[16];
    threadgroup float matG_tile[16][16];
    threadgroup float matQ_tile[16][16];
    threadgroup float matK_tile[16][16];
    threadgroup float matV_tile[16][16];

    // Initialize accumulators cooperatively
    if (tx < 16 && ty < 16) {
        matH_intra_acc[ty][tx] = 0.0f;
    }
    if (tx == 0 && ty < 16) {
        vecN_intra_acc[ty] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute intra-chunk contribution
    uint idx_b_LKV_end = ((idx_b_LQ + 1) * siz_b_LQ) / siz_b_LKV;

    for (uint idx_b_LKV = 0; idx_b_LKV < idx_b_LKV_end; ++idx_b_LKV) {
        // Initialize matG accumulator
        if (tx < 16 && ty < 16) {
            matG_tile[ty][tx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Loop over DHQK blocks to compute matG = matQ @ matK^T
        uint num_DHQK_tiles = (DHQK + siz_b_DHQK - 1) / siz_b_DHQK;
        for (uint idx_b_DHQK = 0; idx_b_DHQK < num_DHQK_tiles; ++idx_b_DHQK) {
            // Load matQ tile cooperatively (siz_b_LQ, siz_b_DHQK)
            for (uint qrow = ty; qrow < siz_b_LQ; qrow += siz_b_LQ) {
                for (uint qcol = tx; qcol < siz_b_DHQK; qcol += siz_b_DHHV) {
                    uint q_global_row = idx_b_NC * L + idx_b_LQ * siz_b_LQ + qrow;
                    uint q_global_col = idx_b_DHQK * siz_b_DHQK + qcol;
                    float q_val = 0.0f;
                    if (q_global_row < S && q_global_col < DHQK) {
                        uint q_idx = idx_b_BNH * str_matQK_B_NH
                                   + q_global_row * str_matQK_S
                                   + q_global_col * str_matQK_DHQK;
                        q_val = matQ[q_idx];
                    }
                    if (qrow < siz_b_LQ && qcol < siz_b_DHQK) {
                        matQ_tile[qrow][qcol] = q_val;
                    }
                }
            }

            // Load matK transposed tile (siz_b_DHQK, siz_b_LKV)
            for (uint krow = ty; krow < siz_b_DHQK; krow += siz_b_LQ) {
                for (uint kcol = tx; kcol < siz_b_LKV; kcol += siz_b_DHHV) {
                    uint k_global_row = idx_b_DHQK * siz_b_DHQK + krow;
                    uint k_global_col = idx_b_NC * L + idx_b_LKV * siz_b_LKV + kcol;
                    float k_val = 0.0f;
                    if (k_global_row < DHQK && k_global_col < S) {
                        uint k_idx = idx_b_BNH * str_matQK_B_NH
                                   + k_global_row * str_matQK_DHQK
                                   + k_global_col * str_matQK_S;
                        k_val = matK[k_idx];
                    }
                    if (krow < siz_b_DHQK && kcol < siz_b_LKV) {
                        matK_tile[krow][kcol] = k_val;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate matG = matQ @ matK (each thread computes one element)
            if (tx < siz_b_LKV && ty < siz_b_LQ) {
                for (uint p = 0; p < siz_b_DHQK; ++p) {
                    matG_tile[ty][tx] = fma(matQ_tile[ty][p], matK_tile[p][tx], matG_tile[ty][tx]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Load vecB_LKV and vecI_LKV
        thread float vecB_LKV[16];
        thread float vecI_LKV[16];
        for (uint i = tx; i < siz_b_LKV; i += siz_b_DHHV) {
            uint b_idx = vecB_base + idx_b_LKV * siz_b_LKV + i;
            uint i_idx = vecI_base + idx_b_LKV * siz_b_LKV + i;
            vecB_LKV[i] = vecB[b_idx];
            vecI_LKV[i] = vecI[i_idx];
        }

        // Compute matDtilde and apply causal mask
        threadgroup float matDtilde[16][16];  // (siz_b_LQ, siz_b_LKV)
        if (tx < siz_b_LKV && ty < siz_b_LQ) {
            float dtilde = vecB_LQ_val[ty] - vecB_LKV[tx] + vecI_LKV[tx];

            // Causal masking
            uint b_kv_offset = idx_b_LKV * siz_b_LKV;
            if (b_kv_offset >= b_q_offset) {
                uint b_q_idx = b_q_offset + ty;
                uint b_kv_idx = b_kv_offset + tx;
                if (b_q_idx < b_kv_idx) {
                    dtilde = -INFINITY;
                }
            }
            matDtilde[ty][tx] = dtilde;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute vecM_new (row-wise max of matDtilde)
        thread float vecM_new_local[16];
        if (ty < siz_b_LQ) {
            float row_max = -INFINITY;
            for (uint j = 0; j < siz_b_LKV; ++j) {
                row_max = fmax(row_max, matDtilde[ty][j]);
            }
            row_max = fmax(row_max, MINIMUM_MAX_VAL);
            row_max = fmax(vecM_old_val[ty], row_max);
            vecM_new_local[ty] = row_max;
            vecM_new_val[ty] = row_max;
        }

        // Compute vecM_ratio
        thread float vecM_ratio[16];
        for (uint i = 0; i < siz_b_LQ; ++i) {
            vecM_ratio[i] = exp(vecM_old_val[i] - vecM_new_local[i]);
        }

        // Compute matD and matS
        threadgroup float matS[16][16];  // (siz_b_LQ, siz_b_LKV)
        if (tx < siz_b_LKV && ty < siz_b_LQ) {
            float d_val = exp(matDtilde[ty][tx] - vecM_new_local[ty]);
            float s_val = matG_tile[ty][tx] * qk_scale * d_val;
            matS[ty][tx] = s_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update vecN_intra_acc (row sum of matS)
        if (tx == 0 && ty < siz_b_LQ) {
            float row_sum = 0.0f;
            for (uint j = 0; j < siz_b_LKV; ++j) {
                row_sum += matS[ty][j];
            }
            vecN_intra_acc[ty] = vecM_ratio[ty] * vecN_intra_acc[ty] + row_sum;
        }

        // Load matV tile (siz_b_LKV, siz_b_DHHV)
        for (uint vrow = ty; vrow < siz_b_LKV; vrow += siz_b_LQ) {
            for (uint vcol = tx; vcol < siz_b_DHHV; vcol += siz_b_DHHV) {
                uint v_global_row = idx_b_NC * L + idx_b_LKV * siz_b_LKV + vrow;
                uint v_global_col = idx_b_DHHV * siz_b_DHHV + vcol;
                float v_val = 0.0f;
                if (v_global_row < S && v_global_col < DHHV) {
                    uint v_idx = idx_b_BNH * str_matHV_B_NH
                               + v_global_row * str_matHV_S
                               + v_global_col * str_matHV_DHHV;
                    v_val = matV[v_idx];
                }
                if (vrow < siz_b_LKV && vcol < siz_b_DHHV) {
                    matV_tile[vrow][vcol] = v_val;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate matH_intra: matS @ matV
        if (tx < siz_b_DHHV && ty < siz_b_LQ) {
            float h_cur = 0.0f;
            for (uint p = 0; p < siz_b_LKV; ++p) {
                h_cur = fma(matS[ty][p], matV_tile[p][tx], h_cur);
            }
            matH_intra_acc[ty][tx] = vecM_ratio[ty] * matH_intra_acc[ty][tx] + h_cur;
        }

        // Update vecM_old for next iteration
        for (uint i = 0; i < siz_b_LQ; ++i) {
            vecM_old_val[i] = vecM_new_val[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute inter-chunk contribution
    // Load scaM_inter_km1
    uint scaM_idx = idx_b_BNH * str_scaMinterstates_B_NH + idx_b_NC;
    float scaM_inter_km1_val = scaMinter_states[scaM_idx];

    // Compute vecM_combine
    thread float vecM_combine_val[16];
    for (uint i = 0; i < siz_b_LQ; ++i) {
        vecM_combine_val[i] = fmax(vecB_LQ_val[i] + scaM_inter_km1_val, vecM_new_val[i]);
    }

    // Compute vecBbar
    thread float vecBbar_val[16];
    for (uint i = 0; i < siz_b_LQ; ++i) {
        vecBbar_val[i] = exp(vecB_LQ_val[i] + scaM_inter_km1_val - vecM_combine_val[i]);
    }

    // Accumulators for inter-chunk
    threadgroup float matH_inter_acc[16][16];  // (siz_b_LQ, siz_b_DHHV)
    threadgroup float vecN_inter_acc[16];       // (siz_b_LQ)

    if (tx < siz_b_DHHV && ty < siz_b_LQ) {
        matH_inter_acc[ty][tx] = 0.0f;
    }
    if (tx == 0 && ty < siz_b_LQ) {
        vecN_inter_acc[ty] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Loop over DHQK blocks
    threadgroup float matQbar_tile[16][16];  // (siz_b_LQ, siz_b_DHQK)
    threadgroup float matC_tile[16][16];     // (siz_b_DHQK, siz_b_DHHV)
    threadgroup float vecN_km1_tile[16];     // (siz_b_DHQK)

    uint num_DHQK_tiles = (DHQK + siz_b_DHQK - 1) / siz_b_DHQK;
    for (uint idx_b_DHQK = 0; idx_b_DHQK < num_DHQK_tiles; ++idx_b_DHQK) {
        // Load matQ and compute matQbar
        for (uint qrow = ty; qrow < siz_b_LQ; qrow += siz_b_LQ) {
            for (uint qcol = tx; qcol < siz_b_DHQK; qcol += siz_b_DHHV) {
                uint q_global_row = idx_b_NC * L + idx_b_LQ * siz_b_LQ + qrow;
                uint q_global_col = idx_b_DHQK * siz_b_DHQK + qcol;
                float q_val = 0.0f;
                if (q_global_row < S && q_global_col < DHQK) {
                    uint q_idx = idx_b_BNH * str_matQK_B_NH
                               + q_global_row * str_matQK_S
                               + q_global_col * str_matQK_DHQK;
                    q_val = matQ[q_idx];
                }
                if (qrow < siz_b_LQ && qcol < siz_b_DHQK) {
                    matQbar_tile[qrow][qcol] = q_val * vecBbar_val[qrow] * qk_scale;
                }
            }
        }

        // Load matC_km1 tile
        for (uint crow = ty; crow < siz_b_DHQK; crow += siz_b_LQ) {
            for (uint ccol = tx; ccol < siz_b_DHHV; ccol += siz_b_DHHV) {
                uint c_global_row = idx_b_DHQK * siz_b_DHQK + crow;
                uint c_global_col = idx_b_DHHV * siz_b_DHHV + ccol;
                float c_val = 0.0f;
                if (c_global_row < DHQK && c_global_col < DHHV) {
                    uint c_idx = idx_b_BNH * str_matCstates_B_NH
                               + idx_b_NC * DHQK * DHHV
                               + c_global_row * str_matCstates_NCDHQK
                               + c_global_col * str_matCstates_DHHV;
                    c_val = matC_states[c_idx];
                }
                if (crow < siz_b_DHQK && ccol < siz_b_DHHV) {
                    matC_tile[crow][ccol] = c_val;
                }
            }
        }

        // Load vecN_km1
        if (tx == 0 && ty < siz_b_DHQK) {
            uint n_idx = idx_b_BNH * str_vecNstates_B_NH
                       + idx_b_NC * DHQK
                       + idx_b_DHQK * siz_b_DHQK + ty;
            float n_val = 0.0f;
            if (idx_b_DHQK * siz_b_DHQK + ty < DHQK) {
                n_val = vecN_states[n_idx];
            }
            vecN_km1_tile[ty] = n_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate matH_inter = matQbar @ matC_km1
        if (tx < siz_b_DHHV && ty < siz_b_LQ) {
            for (uint p = 0; p < siz_b_DHQK; ++p) {
                matH_inter_acc[ty][tx] = fma(matQbar_tile[ty][p], matC_tile[p][tx], matH_inter_acc[ty][tx]);
            }
        }

        // Accumulate vecN_inter = matQbar @ vecN_km1
        if (tx == 0 && ty < siz_b_LQ) {
            for (uint p = 0; p < siz_b_DHQK; ++p) {
                vecN_inter_acc[ty] = fma(matQbar_tile[ty][p], vecN_km1_tile[p], vecN_inter_acc[ty]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Combine intra and inter contributions
    thread float vecM_comb_ratio[16];
    for (uint i = 0; i < siz_b_LQ; ++i) {
        vecM_comb_ratio[i] = exp(vecM_new_val[i] - vecM_combine_val[i]);
    }

    // Compute final output
    if (tx < siz_b_DHHV && ty < siz_b_LQ) {
        float h_num = matH_inter_acc[ty][tx] + vecM_comb_ratio[ty] * matH_intra_acc[ty][tx];
        float n_denom = fmax(fabs(vecN_inter_acc[ty] + vecM_comb_ratio[ty] * vecN_intra_acc[ty]),
                             exp(-vecM_combine_val[ty]));

        float h_out = h_num / (n_denom + EPS);

        // Store matHout
        uint h_global_row = idx_b_NC * L + idx_b_LQ * siz_b_LQ + ty;
        uint h_global_col = idx_b_DHHV * siz_b_DHHV + tx;
        if (h_global_row < S && h_global_col < DHHV) {
            uint h_idx = idx_b_BNH * str_matHV_B_NH
                       + h_global_row * str_matHV_S
                       + h_global_col * str_matHV_DHHV;
            matHout[h_idx] = h_out;
        }
    }

    // Store vecNout and vecMout (only first DHHV tile)
    if (idx_b_DHHV == 0 && tx == 0 && ty < siz_b_LQ) {
        float n_denom = fmax(fabs(vecN_inter_acc[ty] + vecM_comb_ratio[ty] * vecN_intra_acc[ty]),
                             exp(-vecM_combine_val[ty]));

        uint out_idx = idx_b_BNH * str_vecMN_B_NH + idx_b_NC * L + idx_b_LQ * siz_b_LQ + ty;
        if (idx_b_NC * L + idx_b_LQ * siz_b_LQ + ty < S) {
            vecNout[out_idx] = n_denom;
            vecMout[out_idx] = vecM_combine_val[ty];
        }
    }
"""

def mlstm_chunkwise_parallel_fw_Hintra_metal(
    matQ: mx.array,  # (B, NH, S, DHQK)
    matK: mx.array,  # (B, NH, S, DHQK)
    matV: mx.array,  # (B, NH, S, DHHV)
    matC_states: mx.array,  # (B, NH, (NC+1)*DHQK, DHHV)
    vecN_states: mx.array,  # (B, NH, (NC+1)*DHQK)
    scaMinter_states: mx.array,  # (B, NH, NC+1)
    vecI: mx.array,  # (B, NH, NC, L)
    vecB: mx.array,  # (B, NH, NC, L)
    NC: int,
    L: int,
    qk_scale: float,
    siz_b_LQ: int = 8,
    siz_b_LKV: int = 8,
    siz_b_DHQK: int = 8,
    siz_b_DHHV: int = 8,
    eps: float = 1e-6,
    minimum_max_val: float = -10.0,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Metal kernel for parallel forward computation of mLSTM outputs within chunks.

    Returns:
        matHout: (B, NH, S, DHHV)
        vecNout: (B, NH, S)
        vecMout: (B, NH, S)
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[3]

    # Prepare parameters (pack floats as uint32 via reinterpret_cast)
    import struct
    qk_scale_bits = struct.unpack('I', struct.pack('f', qk_scale))[0]
    eps_bits = struct.unpack('I', struct.pack('f', eps))[0]
    min_max_bits = struct.unpack('I', struct.pack('f', minimum_max_val))[0]

    params = mx.array([B, NH, S, DHQK, DHHV, NC, L, siz_b_LQ, siz_b_LKV,
                       siz_b_DHQK, siz_b_DHHV, qk_scale_bits, eps_bits, min_max_bits],
                      dtype=mx.uint32)

    # Prepare strides
    strides = mx.array([
        NH * S * DHQK,  # str_matQK_B_NH
        DHQK,           # str_matQK_S
        1,              # str_matQK_DHQK
        NH * S * DHHV,  # str_matHV_B_NH
        DHHV,           # str_matHV_S
        1,              # str_matHV_DHHV
        (NC + 1) * DHQK * DHHV,  # str_matCstates_B_NH
        DHHV,           # str_matCstates_NCDHQK
        1,              # str_matCstates_DHHV
        (NC + 1) * DHQK,  # str_vecNstates_B_NH
        1,              # str_vecNstates_NCDHQK
        NC + 1,         # str_scaMinterstates_B_NH
        NH * NC * L,    # str_vecBI_B_NH
        L,              # str_vecBI_NC
        1,              # str_vecBI_L
        NH * S,         # str_vecMN_B_NH
        1,              # str_vecMN_S
    ], dtype=mx.uint32)

    # Allocate outputs
    matHout = mx.zeros((B, NH, S, DHHV), dtype=matQ.dtype)
    vecNout = mx.zeros((B, NH, S), dtype=matQ.dtype)
    vecMout = mx.zeros((B, NH, S), dtype=matQ.dtype)

    # Build kernel
    kernel = mx.fast.metal_kernel(
        name="mlstm_parallel_fw_Hintra",
        input_names=["matQ", "matK", "matV", "matC_states", "vecN_states",
                     "scaMinter_states", "vecI", "vecB", "params", "strides"],
        output_names=["matHout", "vecNout", "vecMout"],
        header=_HEADER,
        source=_PARALLEL_FW_HINTRA_SRC,
        ensure_row_contiguous=True,
    )

    # Launch: grid over (DHHV/siz_b_DHHV, L/siz_b_LQ, NC * B*NH)
    num_tiles_DHHV = (DHHV + siz_b_DHHV - 1) // siz_b_DHHV
    num_tiles_LQ = (L + siz_b_LQ - 1) // siz_b_LQ
    grid = (num_tiles_DHHV, num_tiles_LQ, NC * B * NH)
    threadgroup = (siz_b_DHHV, siz_b_LQ, 1)

    outputs = kernel(
        inputs=[matQ, matK, matV, matC_states, vecN_states, scaMinter_states,
                vecI, vecB, params, strides],
        output_shapes=[matHout.shape, vecNout.shape, vecMout.shape],
        output_dtypes=[matQ.dtype, matQ.dtype, matQ.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )

    return outputs


__all__ = ['mlstm_chunkwise_parallel_fw_Hintra_metal']
