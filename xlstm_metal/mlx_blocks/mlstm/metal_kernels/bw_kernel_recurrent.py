#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""Metal kernel for recurrent part of backward pass of the mLSTM chunkwise formulation.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes gradient deltas for C states by backpropagating through chunks
in reverse order (last to first).
"""

import mlx.core as mx
from typing import Tuple, Optional

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_RECURRENT_BW_DC_SRC = r"""
    // Thread and threadgroup indices
    uint idx_b_DHQK = threadgroup_position_in_grid.x;
    uint idx_b_DHHV = threadgroup_position_in_grid.y;
    uint idx_b_NH = threadgroup_position_in_grid.z;

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
    uint siz_b_DHQK = params[7];
    uint siz_b_DHHV = params[8];
    uint save_states_every_nth_chunk = params[9];
    uint USE_LAST_STATE = params[10];

    // Extract floats (reinterpreted from uint32)
    float qk_scale = as_type<float>(params[11]);
    float EPS = as_type<float>(params[12]);

    // Extract strides from strides buffer
    uint str_matQ_B_NH = strides[0];
    uint str_matQ_S = strides[1];
    uint str_matQ_DHQK = strides[2];
    uint str_vecF_B_NH = strides[3];
    uint str_scaM_inter_B_NH = strides[4];
    uint str_scaM_inter_NC = strides[5];
    uint str_vecM_combine_B_NH = strides[6];
    uint str_vecM_combine_S = strides[7];
    uint str_matDeltaH_B_NH = strides[8];
    uint str_matDeltaH_S = strides[9];
    uint str_matDeltaH_DHHV = strides[10];
    uint str_vecN_out_B_NH = strides[11];
    uint str_vecN_out_S = strides[12];
    uint str_matDeltaC_last_B_NH = strides[13];
    uint str_matDeltaC_last_DHQK = strides[14];
    uint str_matDeltaC_last_DHHV = strides[15];
    uint str_matDeltaC_states_B_NH = strides[16];
    uint str_matDeltaC_states_NCDHQK = strides[17];
    uint str_matDeltaC_states_DHHV = strides[18];

    // Threadgroup memory for running deltaC error state (tile, not scalar!)
    threadgroup float matDeltaC_k_val[16][16];  // (siz_b_DHQK, siz_b_DHHV)

    // Initialize to zero cooperatively
    if (tx < siz_b_DHHV && ty < siz_b_DHQK) {
        matDeltaC_k_val[ty][tx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load last state if USE_LAST_STATE
    if (USE_LAST_STATE) {
        uint matDeltaC_row = idx_b_DHQK * siz_b_DHQK + ty;
        uint matDeltaC_col = idx_b_DHHV * siz_b_DHHV + tx;
        if (tx < siz_b_DHHV && ty < siz_b_DHQK && matDeltaC_row < DHQK && matDeltaC_col < DHHV) {
            uint idx = idx_b_NH * str_matDeltaC_last_B_NH
                     + matDeltaC_row * str_matDeltaC_last_DHQK
                     + matDeltaC_col * str_matDeltaC_last_DHHV;
            matDeltaC_k_val[ty][tx] = matDeltaC_last[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Threadgroup memory for temporary arrays
    threadgroup float matQ_k_tile[16][16];       // (siz_b_DHQK, L) transposed
    threadgroup float matDeltaH_k_tile[16][16];  // (L, siz_b_DHHV)
    threadgroup float matQbar_k_tile[16][16];    // (siz_b_DHQK, L)
    threadgroup float vecBbar_k_val[16];         // (L,)
    threadgroup float vecN_out_k_val[16];        // (L,)
    threadgroup float matDeltaC_k_new[16][16];   // Accumulator for new deltaC

    // Thread-local storage for vecF, vecFlogsig
    thread float vecF_local[16];
    thread float vecFlogsig_local[16];

    // Iterate over chunks from last to first (k = NC down to 1)
    for (uint k = NC; k >= 1; --k) {
        // Store matDeltaC_k_val if it's time to save
        if (k % save_states_every_nth_chunk == 0) {
            uint idx_k_save = k / save_states_every_nth_chunk;
            uint matDeltaC_row = idx_b_DHQK * siz_b_DHQK + ty;
            uint matDeltaC_col = idx_b_DHHV * siz_b_DHHV + tx;
            if (tx < siz_b_DHHV && ty < siz_b_DHQK && matDeltaC_row < DHQK && matDeltaC_col < DHHV) {
                uint idx = idx_b_NH * str_matDeltaC_states_B_NH
                         + idx_k_save * DHQK * DHHV
                         + matDeltaC_row * str_matDeltaC_states_NCDHQK
                         + matDeltaC_col * str_matDeltaC_states_DHHV;
                matDeltaC_states[idx] = matDeltaC_k_val[ty][tx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Load vecF for chunk k-1 (0-indexed: chunk index is k-1)
        // Each thread loads one element
        if (ty == 0 && tx < L) {
            uint vec_idx = (k - 1) * L + tx;
            if (vec_idx < S) {
                uint idx = idx_b_NH * str_vecF_B_NH + vec_idx;
                float vecF_elem = vecF[idx];
                vecF_local[tx] = vecF_elem;
                // Compute logsigmoid
                vecFlogsig_local[tx] = log(1.0f / (1.0f + exp(-vecF_elem)));
            } else {
                vecF_local[tx] = 0.0f;
                vecFlogsig_local[tx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute vecB = cumsum(vecFlogsig) and scaG = sum(vecFlogsig)
        // Use shared memory for vecB
        threadgroup float vecB_val[16];
        if (ty == 0 && tx < L) {
            float cumsum = 0.0f;
            for (uint i = 0; i <= tx; ++i) {
                cumsum += vecFlogsig_local[i];
            }
            vecB_val[tx] = cumsum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scaG_k_val = sum of all vecFlogsig
        threadgroup float scaG_k_shared[1];
        if (ty == 0 && tx == 0) {
            float sum_val = 0.0f;
            for (uint i = 0; i < L; ++i) {
                sum_val += vecFlogsig_local[i];
            }
            scaG_k_shared[0] = sum_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scaG_k_val = scaG_k_shared[0];

        // Load scaM_inter_km1, scaM_inter_k
        threadgroup float scaM_inter_km1_shared[1];
        threadgroup float scaM_inter_k_shared[1];
        if (ty == 0 && tx == 0) {
            uint idx_km1 = idx_b_NH * str_scaM_inter_B_NH + (k - 1) * str_scaM_inter_NC;
            uint idx_k = idx_b_NH * str_scaM_inter_B_NH + k * str_scaM_inter_NC;
            scaM_inter_km1_shared[0] = scaM_inter[idx_km1];
            scaM_inter_k_shared[0] = scaM_inter[idx_k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scaM_inter_km1_val = scaM_inter_km1_shared[0];
        float scaM_inter_k_val = scaM_inter_k_shared[0];

        // Load vecM_combine for chunk k-1
        if (ty == 0 && tx < L) {
            uint vec_idx = (k - 1) * L + tx;
            if (vec_idx < S) {
                uint idx = idx_b_NH * str_vecM_combine_B_NH + vec_idx * str_vecM_combine_S;
                vecBbar_k_val[tx] = exp(vecB_val[tx] + scaM_inter_km1_val - vecM_combine[idx]);
            } else {
                vecBbar_k_val[tx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scaGbar_k
        float scaGbar_k_val = exp(scaG_k_val + scaM_inter_km1_val - scaM_inter_k_val);

        // Load matQ chunk k-1 in transposed form: (DHQK, L)
        // Cooperative load
        uint q_chunk_start = (k - 1) * L;
        for (uint tile_l = 0; tile_l < (L + siz_b_DHQK - 1) / siz_b_DHQK; ++tile_l) {
            uint q_row = idx_b_DHQK * siz_b_DHQK + ty;
            uint q_col = tile_l * siz_b_DHQK + tx;
            uint q_seq_idx = q_chunk_start + q_col;

            if (ty < siz_b_DHQK && tx < L && q_col < L && q_row < DHQK && q_seq_idx < S) {
                uint idx = idx_b_NH * str_matQ_B_NH + q_seq_idx * str_matQ_S + q_row * str_matQ_DHQK;
                matQ_k_tile[ty][q_col] = matQ[idx];
            } else if (tx < L && q_col < L) {
                matQ_k_tile[ty][q_col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matQbar_k = matQ_k * vecBbar_k * qk_scale
        if (ty < siz_b_DHQK && tx < L) {
            matQbar_k_tile[ty][tx] = matQ_k_tile[ty][tx] * vecBbar_k_val[tx] * qk_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load vecN_out for chunk k-1
        if (ty == 0 && tx < L) {
            uint vec_idx = (k - 1) * L + tx;
            if (vec_idx < S) {
                uint idx = idx_b_NH * str_vecN_out_B_NH + vec_idx * str_vecN_out_S;
                vecN_out_k_val[tx] = vecN_out[idx];
            } else {
                vecN_out_k_val[tx] = 1.0f;  // Avoid division by zero
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load matDeltaH chunk k-1: (L, DHHV)
        uint deltah_chunk_start = (k - 1) * L;
        for (uint tile_l = 0; tile_l < (L + siz_b_DHHV - 1) / siz_b_DHHV; ++tile_l) {
            uint dh_row = tile_l * siz_b_DHHV + ty;
            uint dh_col = idx_b_DHHV * siz_b_DHHV + tx;
            uint dh_seq_idx = deltah_chunk_start + dh_row;

            if (ty < L && tx < siz_b_DHHV && dh_row < L && dh_col < DHHV && dh_seq_idx < S) {
                uint idx = idx_b_NH * str_matDeltaH_B_NH + dh_seq_idx * str_matDeltaH_S + dh_col * str_matDeltaH_DHHV;
                // Normalize by vecN_out
                matDeltaH_k_tile[dh_row][tx] = matDeltaH[idx] / (vecN_out_k_val[dh_row] + EPS);
            } else if (ty < L && tx < siz_b_DHHV && dh_row < L) {
                matDeltaH_k_tile[dh_row][tx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matDeltaC_km1 = scaGbar_k * matDeltaC_k + matQbar_k @ matDeltaH_k
        // matQbar_k: (siz_b_DHQK, L), matDeltaH_k: (L, siz_b_DHHV)
        // Result: (siz_b_DHQK, siz_b_DHHV)

        // Initialize accumulator
        if (tx < siz_b_DHHV && ty < siz_b_DHQK) {
            matDeltaC_k_new[ty][tx] = scaGbar_k_val * matDeltaC_k_val[ty][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate matQbar @ matDeltaH
        if (tx < siz_b_DHHV && ty < siz_b_DHQK) {
            float sum = 0.0f;
            for (uint l = 0; l < L; ++l) {
                sum += matQbar_k_tile[ty][l] * matDeltaH_k_tile[l][tx];
            }
            matDeltaC_k_new[ty][tx] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Copy new value back to matDeltaC_k_val
        if (tx < siz_b_DHHV && ty < siz_b_DHQK) {
            matDeltaC_k_val[ty][tx] = matDeltaC_k_new[ty][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k == 1) break;  // Prevent underflow
    }

    // Store the first state (after all chunks processed)
    uint matDeltaC_row = idx_b_DHQK * siz_b_DHQK + ty;
    uint matDeltaC_col = idx_b_DHHV * siz_b_DHHV + tx;
    if (tx < siz_b_DHHV && ty < siz_b_DHQK && matDeltaC_row < DHQK && matDeltaC_col < DHHV) {
        uint idx = idx_b_NH * str_matDeltaC_states_B_NH
                 + 0  // First state (index 0)
                 + matDeltaC_row * str_matDeltaC_states_NCDHQK
                 + matDeltaC_col * str_matDeltaC_states_DHHV;
        matDeltaC_states[idx] = matDeltaC_k_val[ty][tx];
    }
"""



def mlstm_chunkwise_recurrent_bw_dC_metal(
    matQ: mx.array,  # (B, NH, S, DHQK)
    vecF: mx.array,  # (B, NH, S)
    scaM_inter: mx.array,  # (B, NH, NC+1)
    vecM_combine: mx.array,  # (B, NH, S)
    matDeltaH: mx.array,  # (B, NH, S, DHHV)
    vecN_out: mx.array,  # (B, NH, S)
    matDeltaC_last: Optional[mx.array],  # (B, NH, DHQK, DHHV)
    NC: int,
    L: int,
    qk_scale: float,
    siz_b_DHQK: int = 16,
    siz_b_DHHV: int = 16,
    save_states_every_nth_chunk: int = 1,
    eps: float = 1e-6,
) -> mx.array:
    """
    Metal kernel for backward recurrent computation of mLSTM gradient deltas.

    Backpropagates through chunks in reverse order (last to first).

    Returns:
        matDeltaC_states: (B, NH, (NC+1)*DHQK, DHHV) - gradient deltas for C states
    """
    import struct

    B, NH, S, DHQK = matQ.shape
    DHHV = matDeltaH.shape[3]

    # Prepare parameter buffer
    USE_LAST_STATE = 1 if matDeltaC_last is not None else 0

    # Pack floats as reinterpreted uint32
    qk_scale_bits = struct.unpack('I', struct.pack('f', qk_scale))[0]
    eps_bits = struct.unpack('I', struct.pack('f', eps))[0]

    params = mx.array([B, NH, S, DHQK, DHHV, NC, L, siz_b_DHQK, siz_b_DHHV,
                       save_states_every_nth_chunk, USE_LAST_STATE,
                       qk_scale_bits, eps_bits], dtype=mx.uint32)

    # Prepare strides buffer
    strides = mx.array([
        NH * S * DHQK,  # str_matQ_B_NH
        DHQK,           # str_matQ_S
        1,              # str_matQ_DHQK
        NH * S,         # str_vecF_B_NH
        NC + 1,         # str_scaM_inter_B_NH
        1,              # str_scaM_inter_NC
        NH * S,         # str_vecM_combine_B_NH
        1,              # str_vecM_combine_S
        NH * S * DHHV,  # str_matDeltaH_B_NH
        DHHV,           # str_matDeltaH_S
        1,              # str_matDeltaH_DHHV
        NH * S,         # str_vecN_out_B_NH
        1,              # str_vecN_out_S
        NH * DHQK * DHHV,  # str_matDeltaC_last_B_NH
        DHHV,           # str_matDeltaC_last_DHQK
        1,              # str_matDeltaC_last_DHHV
        (NC + 1) * DHQK * DHHV,  # str_matDeltaC_states_B_NH
        DHHV,           # str_matDeltaC_states_NCDHQK
        1,              # str_matDeltaC_states_DHHV
    ], dtype=mx.uint32)

    # Allocate output
    matDeltaC_states = mx.zeros((B, NH, (NC + 1) * DHQK, DHHV), dtype=matQ.dtype)

    # Default last state if not provided
    if matDeltaC_last is None:
        matDeltaC_last = mx.zeros((B, NH, DHQK, DHHV), dtype=matQ.dtype)

    # Build kernel
    kernel = mx.fast.metal_kernel(
        name="mlstm_recurrent_bw_dC",
        input_names=["matQ", "vecF", "scaM_inter", "vecM_combine", "matDeltaH",
                     "vecN_out", "matDeltaC_last", "params", "strides"],
        output_names=["matDeltaC_states"],
        header=_HEADER,
        source=_RECURRENT_BW_DC_SRC,
        ensure_row_contiguous=True,
    )

    # Launch: grid over (DHQK/siz_b_DHQK, DHHV/siz_b_DHHV, B*NH)
    num_tiles_DHQK = (DHQK + siz_b_DHQK - 1) // siz_b_DHQK
    num_tiles_DHHV = (DHHV + siz_b_DHHV - 1) // siz_b_DHHV
    grid = (num_tiles_DHQK, num_tiles_DHHV, B * NH)
    threadgroup = (siz_b_DHHV, siz_b_DHQK, 1)

    outputs = kernel(
        inputs=[matQ, vecF, scaM_inter, vecM_combine, matDeltaH,
                vecN_out, matDeltaC_last, params, strides],
        output_shapes=[matDeltaC_states.shape],
        output_dtypes=[matQ.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )

    return outputs[0]


__all__ = ['mlstm_chunkwise_recurrent_bw_dC_metal']
