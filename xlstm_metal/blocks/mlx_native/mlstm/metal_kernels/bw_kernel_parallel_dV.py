#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""Metal kernel for parallel part of backward pass computing dV gradients.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes ∂Loss/∂V using intra-chunk and inter-chunk contributions.
"""

import struct

import mlx.core as mx

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_PARALLEL_BW_DV_SRC = r"""
    // Thread and threadgroup indices
    uint idx_b_DHHV = threadgroup_position_in_grid.x;
    uint idx_b_LKV = threadgroup_position_in_grid.y;
    uint idx_b_NC_BNH = threadgroup_position_in_grid.z;

    uint idx_b_NC = idx_b_NC_BNH % NC;
    uint idx_b_BNH = idx_b_NC_BNH / NC;

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

    // Extract floats
    float qk_scale = as_type<float>(params[11]);
    float EPS = as_type<float>(params[12]);

    // Extract strides
    uint str_matQK_B_NH = strides[0];
    uint str_matQK_S = strides[1];
    uint str_matQK_DHQK = strides[2];
    uint str_matHV_B_NH = strides[3];
    uint str_matHV_S = strides[4];
    uint str_matHV_DHHV = strides[5];
    uint str_vecABI_B_NH = strides[6];
    uint str_vecABI_NC = strides[7];
    uint str_matCstate_B_NH = strides[8];
    uint str_matCstate_NCDHQK = strides[9];
    uint str_matCstate_DHHV = strides[10];
    uint str_vecMN_B_NH = strides[11];
    uint str_vecMN_S = strides[12];

    // Threadgroup memory
    threadgroup float matDeltaV_acc[16][16];  // (siz_b_LKV, siz_b_DHHV)
    threadgroup float matS_trans[16][16];     // (siz_b_LKV, siz_b_LQ)
    threadgroup float matK_tile[16][16];      // (siz_b_LKV, siz_b_DHQK)
    threadgroup float matQ_trans_tile[16][16]; // (siz_b_DHQK, siz_b_LQ)
    threadgroup float matDeltaH_tile[16][16]; // (siz_b_LQ, siz_b_DHHV)
    threadgroup float vecB_LKV[16];
    threadgroup float vecI_LKV[16];
    threadgroup float vecB_LQ[16];
    threadgroup float vecM_out_LQ[16];
    threadgroup float vecN_out_LQ[16];

    // Initialize accumulator
    if (tx < siz_b_DHHV && ty < siz_b_LKV) {
        matDeltaV_acc[ty][tx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load vecB_LKV and vecI_LKV for this chunk
    if (ty == 0 && tx < siz_b_LKV) {
        uint vec_idx = idx_b_LKV * siz_b_LKV + tx;
        if (vec_idx < L) {
            uint idx_b = idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC + vec_idx;
            vecB_LKV[tx] = vecB[idx_b];
            vecI_LKV[tx] = vecI[idx_b];
        } else {
            vecB_LKV[tx] = 0.0f;
            vecI_LKV[tx] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load scaMinter_k for inter-chunk contribution
    threadgroup float scaMinter_k_shared[1];
    if (ty == 0 && tx == 0) {
        uint idx = idx_b_BNH * (NC + 1) + (idx_b_NC + 1);
        scaMinter_k_shared[0] = scaMstate_all[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scaMinter_k_val = scaMinter_k_shared[0];

    // Load vecA for this chunk and compute vecAbar
    threadgroup float vecAbar[16];
    if (ty == 0 && tx < siz_b_LKV) {
        uint vec_idx = idx_b_LKV * siz_b_LKV + tx;
        if (vec_idx < L) {
            uint idx_a = idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC + vec_idx;
            float vecA_val = vecA[idx_a];
            vecAbar[tx] = exp(vecA_val - scaMinter_k_val);
        } else {
            vecAbar[tx] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // For causal masking
    uint b_kv_offset_start = idx_b_LKV * siz_b_LKV;
    uint b_kv_offset_end = (idx_b_LKV + 1) * siz_b_LKV;

    //! Intra-chunk contribution
    // Loop over siz_b_LQ blocks (only upper triangular)
    uint idx_b_LQ_start = (idx_b_LKV * siz_b_LKV) / siz_b_LQ;
    uint idx_b_LQ_end = (L + siz_b_LQ - 1) / siz_b_LQ;

    for (uint idx_b_LQ = idx_b_LQ_start; idx_b_LQ < idx_b_LQ_end; ++idx_b_LQ) {
        // Compute matS^T block (siz_b_LKV, siz_b_LQ)
        if (tx < siz_b_LQ && ty < siz_b_LKV) {
            matS_trans[ty][tx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Loop over siz_b_DHQK blocks to compute matS = matK @ matQ^T
        for (uint idx_b_DHQK = 0; idx_b_DHQK < (DHQK + siz_b_DHQK - 1) / siz_b_DHQK; ++idx_b_DHQK) {
            // Load matK (siz_b_LKV, siz_b_DHQK)
            uint k_row = idx_b_LKV * siz_b_LKV + ty;
            uint k_col = idx_b_DHQK * siz_b_DHQK + tx;
            uint k_seq_idx = idx_b_NC * L + k_row;

            if (ty < siz_b_LKV && tx < siz_b_DHQK && k_row < L && k_col < DHQK && k_seq_idx < S) {
                uint idx = idx_b_BNH * str_matQK_B_NH + k_seq_idx * str_matQK_S + k_col * str_matQK_DHQK;
                matK_tile[ty][tx] = matK[idx];
            } else if (ty < siz_b_LKV && tx < siz_b_DHQK) {
                matK_tile[ty][tx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load matQ^T (siz_b_DHQK, siz_b_LQ)
            uint q_row = idx_b_DHQK * siz_b_DHQK + ty;
            uint q_col = idx_b_LQ * siz_b_LQ + tx;
            uint q_seq_idx = idx_b_NC * L + q_col;

            if (ty < siz_b_DHQK && tx < siz_b_LQ && q_row < DHQK && q_col < L && q_seq_idx < S) {
                uint idx = idx_b_BNH * str_matQK_B_NH + q_seq_idx * str_matQK_S + q_row * str_matQK_DHQK;
                matQ_trans_tile[ty][tx] = matQ[idx];
            } else if (ty < siz_b_DHQK && tx < siz_b_LQ) {
                matQ_trans_tile[ty][tx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate matS^T = matK @ matQ^T
            if (tx < siz_b_LQ && ty < siz_b_LKV) {
                float sum = 0.0f;
                for (uint k = 0; k < siz_b_DHQK; ++k) {
                    sum += matK_tile[ty][k] * matQ_trans_tile[k][tx];
                }
                matS_trans[ty][tx] += sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Load vecB_LQ
        if (ty == 0 && tx < siz_b_LQ) {
            uint vec_idx = idx_b_LQ * siz_b_LQ + tx;
            if (vec_idx < L) {
                uint idx_b = idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC + vec_idx;
                vecB_LQ[tx] = vecB[idx_b];
            } else {
                vecB_LQ[tx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matDtilde and apply causal mask
        threadgroup float matD_trans[16][16];
        if (tx < siz_b_LQ && ty < siz_b_LKV) {
            float matDtilde_val = vecB_LQ[tx] - vecB_LKV[ty] + vecI_LKV[ty];

            // Causal masking
            uint b_q_offset = idx_b_LQ * siz_b_LQ;
            if (b_kv_offset_end >= b_q_offset) {
                uint q_idx = b_q_offset + tx;
                uint kv_idx = b_kv_offset_start + ty;
                if (q_idx < kv_idx) {
                    matDtilde_val = -INFINITY;
                }
            }

            matD_trans[ty][tx] = matDtilde_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load vecM_out
        if (ty == 0 && tx < siz_b_LQ) {
            uint vec_idx = idx_b_NC * L + idx_b_LQ * siz_b_LQ + tx;
            if (vec_idx < S) {
                uint idx = idx_b_BNH * str_vecMN_B_NH + vec_idx * str_vecMN_S;
                vecM_out_LQ[tx] = vecM_out[idx];
            } else {
                vecM_out_LQ[tx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matD_trans = exp(matDtilde - vecM_out)
        if (tx < siz_b_LQ && ty < siz_b_LKV) {
            matD_trans[ty][tx] = exp(matD_trans[ty][tx] - vecM_out_LQ[tx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matSbar_trans = matS_trans * qk_scale * matD_trans
        threadgroup float matSbar_trans[16][16];
        if (tx < siz_b_LQ && ty < siz_b_LKV) {
            matSbar_trans[ty][tx] = matS_trans[ty][tx] * qk_scale * matD_trans[ty][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load matDeltaH_out (siz_b_LQ, siz_b_DHHV)
        uint dh_row = idx_b_LQ * siz_b_LQ + ty;
        uint dh_col = idx_b_DHHV * siz_b_DHHV + tx;
        uint dh_seq_idx = idx_b_NC * L + dh_row;

        if (ty < siz_b_LQ && tx < siz_b_DHHV && dh_row < L && dh_col < DHHV && dh_seq_idx < S) {
            uint idx = idx_b_BNH * str_matHV_B_NH + dh_seq_idx * str_matHV_S + dh_col * str_matHV_DHHV;
            matDeltaH_tile[ty][tx] = matDeltaH_out[idx];
        } else if (ty < siz_b_LQ && tx < siz_b_DHHV) {
            matDeltaH_tile[ty][tx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load vecN_out
        if (ty == 0 && tx < siz_b_LQ) {
            uint vec_idx = idx_b_NC * L + idx_b_LQ * siz_b_LQ + tx;
            if (vec_idx < S) {
                uint idx = idx_b_BNH * str_vecMN_B_NH + vec_idx * str_vecMN_S;
                vecN_out_LQ[tx] = vecN_out[idx];
            } else {
                vecN_out_LQ[tx] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Normalize matDeltaH by vecN_out
        if (ty < siz_b_LQ && tx < siz_b_DHHV) {
            matDeltaH_tile[ty][tx] /= (vecN_out_LQ[ty] + EPS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate matDeltaV = matSbar_trans @ matDeltaH
        if (tx < siz_b_DHHV && ty < siz_b_LKV) {
            float sum = 0.0f;
            for (uint l = 0; l < siz_b_LQ; ++l) {
                sum += matSbar_trans[ty][l] * matDeltaH_tile[l][tx];
            }
            matDeltaV_acc[ty][tx] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    //! Inter-chunk contribution (computed only on first iteration)
    if (idx_b_LQ_start == (idx_b_LKV * siz_b_LKV) / siz_b_LQ) {
        // Loop over DHQK tiles
        for (uint idx_b_DHQK = 0; idx_b_DHQK < (DHQK + siz_b_DHQK - 1) / siz_b_DHQK; ++idx_b_DHQK) {
            // Load matK and compute matKbar
            uint k_row = idx_b_LKV * siz_b_LKV + ty;
            uint k_col = idx_b_DHQK * siz_b_DHQK + tx;
            uint k_seq_idx = idx_b_NC * L + k_row;

            threadgroup float matKbar_tile[16][16];
            if (ty < siz_b_LKV && tx < siz_b_DHQK && k_row < L && k_col < DHQK && k_seq_idx < S) {
                uint idx = idx_b_BNH * str_matQK_B_NH + k_seq_idx * str_matQK_S + k_col * str_matQK_DHQK;
                matKbar_tile[ty][tx] = matK[idx] * vecAbar[ty];
            } else if (ty < siz_b_LKV && tx < siz_b_DHQK) {
                matKbar_tile[ty][tx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load matDeltaC (siz_b_DHQK, siz_b_DHHV)
            threadgroup float matDeltaC_tile[16][16];
            uint dc_row = idx_b_DHQK * siz_b_DHQK + ty;
            uint dc_col = idx_b_DHHV * siz_b_DHHV + tx;

            if (ty < siz_b_DHQK && tx < siz_b_DHHV && dc_row < DHQK && dc_col < DHHV) {
                uint idx = idx_b_BNH * str_matCstate_B_NH + (idx_b_NC + 1) * DHQK * DHHV
                         + dc_row * str_matCstate_NCDHQK + dc_col * str_matCstate_DHHV;
                matDeltaC_tile[ty][tx] = matDeltaC_states[idx];
            } else if (ty < siz_b_DHQK && tx < siz_b_DHHV) {
                matDeltaC_tile[ty][tx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate matDeltaV_inter = matKbar @ matDeltaC
            if (tx < siz_b_DHHV && ty < siz_b_LKV) {
                float sum = 0.0f;
                for (uint k = 0; k < siz_b_DHQK; ++k) {
                    sum += matKbar_tile[ty][k] * matDeltaC_tile[k][tx];
                }
                matDeltaV_acc[ty][tx] += sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store matDeltaV
    uint dv_row = idx_b_NC * L + idx_b_LKV * siz_b_LKV + ty;
    uint dv_col = idx_b_DHHV * siz_b_DHHV + tx;

    if (ty < siz_b_LKV && tx < siz_b_DHHV && dv_row < S && dv_col < DHHV) {
        uint idx = idx_b_BNH * str_matHV_B_NH + dv_row * str_matHV_S + dv_col * str_matHV_DHHV;
        matDeltaV[idx] = matDeltaV_acc[ty][tx];
    }
"""


def mlstm_chunkwise_parallel_bw_dV_metal(
    matQ: mx.array,  # (B, NH, S, DHQK)
    matK: mx.array,  # (B, NH, S, DHQK)
    matV: mx.array,  # (B, NH, S, DHHV)
    vecI: mx.array,  # (B, NH, NC, L)
    vecB: mx.array,  # (B, NH, NC, L)
    vecA: mx.array,  # (B, NH, NC, L)
    matCstate_all: mx.array,  # (B, NH, (NC+1)*DHQK, DHHV)
    vecNstate_all: mx.array,  # (B, NH, (NC+1)*DHQK)
    scaMstate_all: mx.array,  # (B, NH, NC+1)
    vecN_out: mx.array,  # (B, NH, S)
    vecM_out: mx.array,  # (B, NH, S)
    matDeltaH_out: mx.array,  # (B, NH, S, DHHV)
    matDeltaC_states: mx.array,  # (B, NH, (NC+1)*DHQK, DHHV)
    NC: int,
    L: int,
    qk_scale: float,
    siz_b_LQ: int = 8,
    siz_b_LKV: int = 8,
    siz_b_DHQK: int = 8,
    siz_b_DHHV: int = 8,
    eps: float = 1e-6,
) -> mx.array:
    """
    Metal kernel for parallel backward computation of dV gradients.

    Returns:
        matDeltaV: (B, NH, S, DHHV) - gradients w.r.t. V
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[3]

    # Pack floats
    qk_scale_bits = struct.unpack('I', struct.pack('f', qk_scale))[0]
    eps_bits = struct.unpack('I', struct.pack('f', eps))[0]

    params = mx.array([B, NH, S, DHQK, DHHV, NC, L, siz_b_LQ, siz_b_LKV,
                       siz_b_DHQK, siz_b_DHHV, qk_scale_bits, eps_bits],
                      dtype=mx.uint32)

    strides = mx.array([
        NH * S * DHQK,  # str_matQK_B_NH
        DHQK,           # str_matQK_S
        1,              # str_matQK_DHQK
        NH * S * DHHV,  # str_matHV_B_NH
        DHHV,           # str_matHV_S
        1,              # str_matHV_DHHV
        NH * NC * L,    # str_vecABI_B_NH
        L,              # str_vecABI_NC
        (NC + 1) * DHQK * DHHV,  # str_matCstate_B_NH
        DHHV,           # str_matCstate_NCDHQK
        1,              # str_matCstate_DHHV
        NH * S,         # str_vecMN_B_NH
        1,              # str_vecMN_S
    ], dtype=mx.uint32)

    # Allocate output
    matDeltaV = mx.zeros((B, NH, S, DHHV), dtype=matQ.dtype)

    # Build kernel
    kernel = mx.fast.metal_kernel(name="mlstm_parallel_bw_dV",
                                  input_names=["matQ", "matK", "matV", "vecI", "vecB", "vecA",
                                               "matCstate_all", "vecNstate_all", "scaMstate_all",
                                               "vecN_out", "vecM_out", "matDeltaH_out", "matDeltaC_states",
                                               "params", "strides"], output_names=["matDeltaV"], header=_HEADER,
                                  source=_PARALLEL_BW_DV_SRC)

    # Launch: grid over (DHHV/siz_b_DHHV, L/siz_b_LKV, NC * B*NH)
    num_tiles_DHHV = (DHHV + siz_b_DHHV - 1) // siz_b_DHHV
    num_tiles_LKV = (L + siz_b_LKV - 1) // siz_b_LKV
    grid = (num_tiles_DHHV, num_tiles_LKV, NC * B * NH)
    threadgroup = (siz_b_DHHV, siz_b_LKV, 1)

    outputs = kernel(
        inputs=[matQ, matK, matV, vecI, vecB, vecA,
                matCstate_all, vecNstate_all, scaMstate_all,
                vecN_out, vecM_out, matDeltaH_out, matDeltaC_states,
                params, strides],
        output_shapes=[matDeltaV.shape],
        output_dtypes=[matQ.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )

    return outputs[0]


__all__ = ['mlstm_chunkwise_parallel_bw_dV_metal']
