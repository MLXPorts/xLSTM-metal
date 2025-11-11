#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""Metal kernel for parallel part of backward pass computing dQ gradients.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes ∂Loss/∂Q using intra-chunk and inter-chunk contributions.
"""

import struct

import mlx.core as mx

HEADER = """#include <metal_stdlib>
using namespace metal;
"""

PARALLEL_BW_DQ_SRC = r"""
    // Thread and threadgroup indices
    uint idx_b_DHQK = threadgroup_position_in_grid.x;
    uint idx_b_LQ = threadgroup_position_in_grid.y;
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
    threadgroup float matDeltaQ_acc[16][16];      // (siz_b_LQ, siz_b_DHQK)
    threadgroup float matDeltaSbar[16][16];       // (siz_b_LQ, siz_b_LKV)
    threadgroup float matDeltaH_tile[16][16];     // (siz_b_LQ, siz_b_DHHV)
    threadgroup float matV_trans_tile[16][16];    // (siz_b_DHHV, siz_b_LKV)
    threadgroup float matK_tile[16][16];          // (siz_b_LKV, siz_b_DHQK)
    threadgroup float vecB_LQ[16];
    threadgroup float vecI_LKV[16];
    threadgroup float vecB_LKV[16];
    threadgroup float vecM_out_LQ[16];
    threadgroup float vecN_out_LQ[16];

    // Initialize accumulator
    if (tx < siz_b_DHQK && ty < siz_b_LQ) {
        matDeltaQ_acc[ty][tx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load vecB_LQ for this chunk
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

    // Load vecN_out and vecM_out for this LQ block
    if (ty == 0 && tx < siz_b_LQ) {
        uint vec_idx = idx_b_NC * L + idx_b_LQ * siz_b_LQ + tx;
        if (vec_idx < S) {
            uint idx = idx_b_BNH * str_vecMN_B_NH + vec_idx * str_vecMN_S;
            vecN_out_LQ[tx] = vecN_out[idx];
            vecM_out_LQ[tx] = vecM_out[idx];
        } else {
            vecN_out_LQ[tx] = 1.0f;
            vecM_out_LQ[tx] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load scaMinter_km1 for inter-chunk contribution
    threadgroup float scaMinter_km1_shared[1];
    if (ty == 0 && tx == 0) {
        uint idx = idx_b_BNH * (NC + 1) + idx_b_NC;  // km1 not k
        scaMinter_km1_shared[0] = scaMstate_all[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scaMinter_km1_val = scaMinter_km1_shared[0];

    // Load vecB for this chunk and compute vecBbar
    threadgroup float vecBbar[16];
    if (ty == 0 && tx < siz_b_LQ) {
        uint vec_idx = idx_b_LQ * siz_b_LQ + tx;
        if (vec_idx < L) {
            uint idx_b = idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC + vec_idx;
            float vecB_val = vecB[idx_b];
            vecBbar[tx] = exp(vecB_val - scaMinter_km1_val);
        } else {
            vecBbar[tx] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // For causal masking
    uint b_q_offset = idx_b_LQ * siz_b_LQ;

    //! Intra-chunk contribution
    // Loop over siz_b_LKV blocks (only lower triangular)
    uint idx_b_LKV_end = ((idx_b_LQ + 1) * siz_b_LQ + siz_b_LKV - 1) / siz_b_LKV;

    for (uint idx_b_LKV = 0; idx_b_LKV < idx_b_LKV_end; ++idx_b_LKV) {
        // Compute matDeltaSbar block (siz_b_LQ, siz_b_LKV)
        if (tx < siz_b_LKV && ty < siz_b_LQ) {
            matDeltaSbar[ty][tx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Loop over siz_b_DHHV blocks to compute matDeltaSbar = matDeltaH @ matV_trans
        for (uint idx_b_DHHV = 0; idx_b_DHHV < (DHHV + siz_b_DHHV - 1) / siz_b_DHHV; ++idx_b_DHHV) {
            // Load matDeltaH (siz_b_LQ, siz_b_DHHV)
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

            //! Inter-chunk contribution (computed only on first iteration)
            if (idx_b_LKV == 0) {
                // Load matC_km1_trans (siz_b_DHHV, siz_b_DHQK)
                threadgroup float matC_km1_trans_tile[16][16];
                uint c_row = idx_b_DHHV * siz_b_DHHV + ty;
                uint c_col = idx_b_DHQK * siz_b_DHQK + tx;

                if (ty < siz_b_DHHV && tx < siz_b_DHQK && c_row < DHHV && c_col < DHQK) {
                    // Load transposed by swapping strides
                    uint idx = idx_b_BNH * str_matCstate_B_NH + idx_b_NC * DHQK * DHHV
                             + c_col * str_matCstate_NCDHQK + c_row * str_matCstate_DHHV;
                    matC_km1_trans_tile[ty][tx] = matCstate_all[idx];
                } else if (ty < siz_b_DHHV && tx < siz_b_DHQK) {
                    matC_km1_trans_tile[ty][tx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute matDeltaQbar_inter = (matDeltaH @ matC_km1_trans) / vecN_out
                if (tx < siz_b_DHQK && ty < siz_b_LQ) {
                    float sum = 0.0f;
                    for (uint k = 0; k < siz_b_DHHV; ++k) {
                        sum += matDeltaH_tile[ty][k] * matC_km1_trans_tile[k][tx];
                    }
                    // Normalize and apply vecBbar gating
                    sum /= (vecN_out_LQ[ty] + EPS);
                    matDeltaQ_acc[ty][tx] += sum * vecBbar[ty] * qk_scale;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Load matV_trans (siz_b_DHHV, siz_b_LKV)
            uint v_row = idx_b_DHHV * siz_b_DHHV + ty;
            uint v_col = idx_b_LKV * siz_b_LKV + tx;
            uint v_seq_idx = idx_b_NC * L + v_col;

            if (ty < siz_b_DHHV && tx < siz_b_LKV && v_row < DHHV && v_col < L && v_seq_idx < S) {
                // Load transposed by swapping indices
                uint idx = idx_b_BNH * str_matHV_B_NH + v_seq_idx * str_matHV_S + v_row * str_matHV_DHHV;
                matV_trans_tile[ty][tx] = matV[idx];
            } else if (ty < siz_b_DHHV && tx < siz_b_LKV) {
                matV_trans_tile[ty][tx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate matDeltaSbar = matDeltaH @ matV_trans
            if (tx < siz_b_LKV && ty < siz_b_LQ) {
                float sum = 0.0f;
                for (uint k = 0; k < siz_b_DHHV; ++k) {
                    sum += matDeltaH_tile[ty][k] * matV_trans_tile[k][tx];
                }
                matDeltaSbar[ty][tx] += sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Load vecI_LKV and vecB_LKV
        if (ty == 0 && tx < siz_b_LKV) {
            uint vec_idx = idx_b_LKV * siz_b_LKV + tx;
            if (vec_idx < L) {
                uint idx_b = idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC + vec_idx;
                vecI_LKV[tx] = vecI[idx_b];
                vecB_LKV[tx] = vecB[idx_b];
            } else {
                vecI_LKV[tx] = 0.0f;
                vecB_LKV[tx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matDtilde and apply causal mask
        threadgroup float matD[16][16];
        if (tx < siz_b_LKV && ty < siz_b_LQ) {
            float matDtilde_val = vecB_LQ[ty] - vecB_LKV[tx] + vecI_LKV[tx];

            // Causal masking
            uint b_kv_offset = idx_b_LKV * siz_b_LKV;
            if (b_kv_offset >= b_q_offset) {
                uint q_idx = b_q_offset + ty;
                uint kv_idx = b_kv_offset + tx;
                if (q_idx < kv_idx) {
                    matDtilde_val = -INFINITY;
                }
            }

            matD[ty][tx] = matDtilde_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matD = exp(matDtilde - vecM_out)
        if (tx < siz_b_LKV && ty < siz_b_LQ) {
            matD[ty][tx] = exp(matD[ty][tx] - vecM_out_LQ[ty]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Normalize matDeltaSbar by vecN_out and compute matDeltaS
        threadgroup float matDeltaS[16][16];
        if (tx < siz_b_LKV && ty < siz_b_LQ) {
            float normalized = matDeltaSbar[ty][tx] / (vecN_out_LQ[ty] + EPS);
            matDeltaS[ty][tx] = normalized * qk_scale * matD[ty][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        // Accumulate matDeltaQ = matDeltaS @ matK
        if (tx < siz_b_DHQK && ty < siz_b_LQ) {
            float sum = 0.0f;
            for (uint l = 0; l < siz_b_LKV; ++l) {
                sum += matDeltaS[ty][l] * matK_tile[l][tx];
            }
            matDeltaQ_acc[ty][tx] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store matDeltaQ
    uint dq_row = idx_b_NC * L + idx_b_LQ * siz_b_LQ + ty;
    uint dq_col = idx_b_DHQK * siz_b_DHQK + tx;

    if (ty < siz_b_LQ && tx < siz_b_DHQK && dq_row < S && dq_col < DHQK) {
        uint idx = idx_b_BNH * str_matQK_B_NH + dq_row * str_matQK_S + dq_col * str_matQK_DHQK;
        matDeltaQ[idx] = matDeltaQ_acc[ty][tx];
    }
"""


def mlstm_chunkwise_parallel_bw_dQ_metal(
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
    Metal kernel for parallel backward computation of dQ gradients.

    Returns:
        matDeltaQ: (B, NH, S, DHQK) - gradients w.r.t. Q
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
        DHQK,  # str_matQK_S
        1,  # str_matQK_DHQK
        NH * S * DHHV,  # str_matHV_B_NH
        DHHV,  # str_matHV_S
        1,  # str_matHV_DHHV
        NH * NC * L,  # str_vecABI_B_NH
        L,  # str_vecABI_NC
        (NC + 1) * DHQK * DHHV,  # str_matCstate_B_NH
        DHHV,  # str_matCstate_NCDHQK
        1,  # str_matCstate_DHHV
        NH * S,  # str_vecMN_B_NH
        1,  # str_vecMN_S
    ], dtype=mx.uint32)

    # Allocate output
    matDeltaQ = mx.zeros((B, NH, S, DHQK), dtype=matQ.dtype)

    # Build kernel
    kernel = mx.fast.metal_kernel(name="mlstm_parallel_bw_dQ",
                                  input_names=["matQ", "matK", "matV", "vecI", "vecB", "vecA",
                                               "matCstate_all", "vecNstate_all", "scaMstate_all",
                                               "vecN_out", "vecM_out", "matDeltaH_out", "matDeltaC_states",
                                               "params", "strides"], output_names=["matDeltaQ"], header=HEADER,
                                  source=PARALLEL_BW_DQ_SRC)

    # Launch: grid over (DHQK/siz_b_DHQK, L/siz_b_LQ, NC * B*NH)
    num_tiles_DHQK = (DHQK + siz_b_DHQK - 1) // siz_b_DHQK
    num_tiles_LQ = (L + siz_b_LQ - 1) // siz_b_LQ
    grid = (num_tiles_DHQK, num_tiles_LQ, NC * B * NH)
    threadgroup = (siz_b_DHQK, siz_b_LQ, 1)

    outputs = kernel(
        inputs=[matQ, matK, matV, vecI, vecB, vecA,
                matCstate_all, vecNstate_all, scaMstate_all,
                vecN_out, vecM_out, matDeltaH_out, matDeltaC_states,
                params, strides],
        output_shapes=[matDeltaQ.shape],
        output_dtypes=[matQ.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )

    return outputs[0]


__all__ = ['mlstm_chunkwise_parallel_bw_dQ_metal']
