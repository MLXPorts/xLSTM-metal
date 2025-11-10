#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""Metal kernel for recurrent part of forward pass of the mLSTM chunkwise formulation.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes inter-chunk states (C, n, m) sequentially across chunks.
Each threadgroup processes a (siz_b_DHQK, siz_b_DHHV) tile of the C matrix.
"""

from typing import Tuple, Optional

import mlx.core as mx

HEADER = """#include <metal_stdlib>
using namespace metal;
"""

RECURRENT_FW_C_SRC = r"""
    // Thread and threadgroup indices (Triton: tl.program_id)
    uint idx_b_DHQK = threadgroup_position_in_grid.x;
    uint idx_b_DHHV = threadgroup_position_in_grid.y;
    uint idx_b_BNH = threadgroup_position_in_grid.z;

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
    uint USE_INITIAL_STATE = params[10];
    uint USE_DBG = params[11];

    // Extract strides from strides buffer
    uint str_matK_B_NH = strides[0];
    uint str_matK_S = strides[1];
    uint str_matK_DHQK = strides[2];
    uint str_matV_B_NH = strides[3];
    uint str_matV_S = strides[4];
    uint str_matV_DHHV = strides[5];
    uint str_vecFI_B_NH = strides[6];
    uint str_matCstates_B_NH = strides[7];
    uint str_matCstates_NCDHQK = strides[8];
    uint str_matCstates_DHHV = strides[9];
    uint str_vecNstates_B_NH = strides[10];
    uint str_vecNstates_NCDHQK = strides[11];
    uint str_scaMinterstates_B_NH = strides[12];
    uint str_scaMinterstates_NC = strides[13];
    uint str_matCinitial_B_NH = strides[14];
    uint str_matCinitial_DHQK = strides[15];
    uint str_matCinitial_DHHV = strides[16];
    uint str_vecNinitial_B_NH = strides[17];
    uint str_vecNinitial_DHQK = strides[18];
    uint str_scaMinterinitial_B_NH = strides[19];

    // Threadgroup memory for running states (tiles, not scalars!)
    threadgroup float matC_k_val[16][16];  // Will be sized by siz_b_DHQK x siz_b_DHHV
    threadgroup float vecN_k_val[16];      // Will be sized by siz_b_DHQK
    threadgroup float scaMinter_k_val_shared[1];  // Shared scalar across threadgroup

    // Initialize to zero cooperatively
    if (tx < siz_b_DHHV && ty < siz_b_DHQK) {
        matC_k_val[ty][tx] = 0.0f;
    }
    if (tx == 0 && ty < siz_b_DHQK) {
        vecN_k_val[ty] = 0.0f;
    }
    if (tx == 0 && ty == 0) {
        scaMinter_k_val_shared[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load initial states if USE_INITIAL_STATE
    if (USE_INITIAL_STATE) {
        // Load matC_initial tile (siz_b_DHQK, siz_b_DHHV)
        uint matC_row = idx_b_DHQK * siz_b_DHQK + ty;
        uint matC_col = idx_b_DHHV * siz_b_DHHV + tx;
        if (tx < siz_b_DHHV && ty < siz_b_DHQK && matC_row < DHQK && matC_col < DHHV) {
            uint idx = idx_b_BNH * str_matCinitial_B_NH
                     + matC_row * str_matCinitial_DHQK
                     + matC_col * str_matCinitial_DHHV;
            matC_k_val[ty][tx] = matC_initial[idx];
        }

        // Load vecN_initial chunk (siz_b_DHQK,)
        if (tx == 0 && ty < siz_b_DHQK) {
            uint vecN_idx = idx_b_DHQK * siz_b_DHQK + ty;
            if (vecN_idx < DHQK) {
                uint idx = idx_b_BNH * str_vecNinitial_B_NH + vecN_idx * str_vecNinitial_DHQK;
                vecN_k_val[ty] = vecN_initial[idx];
            }
        }

        // Load scaMinter_initial (scalar)
        if (tx == 0 && ty == 0) {
            uint idx = idx_b_BNH * str_scaMinterinitial_B_NH;
            scaMinter_k_val_shared[0] = scaMinter_initial[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Threadgroup shared memory for vecFlogsig_masked and vecA (CRITICAL: must be shared across threads!)
    threadgroup float vecFlogsig_masked_shared[64];
    threadgroup float vecA_shared[64];
    threadgroup float vecI_shared[64];

    // Iterate over chunks
    for (uint k = 0; k < NC; ++k) {
        uint tid_linear = ty * siz_b_DHHV + tx; // Declare tid_linear here
        uint num_threads = siz_b_DHQK * siz_b_DHHV; // Declare num_threads here

        // Store states every nth chunk
        if (k % save_states_every_nth_chunk == 0) {
            uint idx_k_save = k / save_states_every_nth_chunk;

            // Store matC_k_val tile
            uint matC_row = idx_b_DHQK * siz_b_DHQK + ty;
            uint matC_col = idx_b_DHHV * siz_b_DHHV + tx;
            if (tx < siz_b_DHHV && ty < siz_b_DHQK && matC_row < DHQK && matC_col < DHHV) {
                uint idx = idx_b_BNH * str_matCstates_B_NH
                         + idx_k_save * DHQK * DHHV
                         + matC_row * str_matCstates_NCDHQK
                         + matC_col * str_matCstates_DHHV;
                matC_states[idx] = matC_k_val[ty][tx];
            }

            // Store vecN_k_val (only idx_b_DHHV == 0)
            if (idx_b_DHHV == 0 && tx == 0 && ty < siz_b_DHQK) {
                uint vecN_idx = idx_b_DHQK * siz_b_DHQK + ty;
                if (vecN_idx < DHQK) {
                    uint idx = idx_b_BNH * str_vecNstates_B_NH
                             + idx_k_save * DHQK
                             + vecN_idx * str_vecNstates_NCDHQK;
                    vecN_states[idx] = vecN_k_val[ty];
                }
            }

            // Store scaMinter_k_val (only first thread of first tile)
            if (idx_b_DHQK == 0 && idx_b_DHHV == 0 && tx == 0 && ty == 0) {
                uint idx = idx_b_BNH * str_scaMinterstates_B_NH + idx_k_save;
                scaMinter_states[idx] = scaMinter_k_val_shared[0];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // STEP 1+2: Compute vecA (B_last - B + I), vecFlogsig_masked for debug, and prepare shared buffers
        // Single-thread deterministic implementation following Transformers canonical formula.
        if (tx == 0 && ty == 0) {
            float tail_sum = 0.0f; // sum of logsig(f[j]) for j > i
            for (int i = int(L) - 1; i >= 0; --i) {
                // load gates
                uint f_idx = idx_b_BNH * str_vecFI_B_NH + k * L + (uint)i;
                float f_val = vecF[f_idx];
                // logsigmoid(x) = -log(1 + exp(-x))
                float f_logsig = -log(1.0f + exp(-f_val));

                // I gate
                uint i_idx = idx_b_BNH * str_vecFI_B_NH + k * L + (uint)i;
                float i_val = vecI[i_idx];
                // A[i] = sum_{j>i} logsig(F[j]) + I[i]
                vecA_shared[i] = tail_sum + i_val;

                // For debugging: masked logsig (exclude current, last=0)
                vecFlogsig_masked_shared[i] = (i < int(L) - 1) ? (-log(1.0f + exp(-vecF[(uint)(i+1) + k*L + idx_b_BNH * str_vecFI_B_NH]))) : 0.0f;

                // advance tail sum to include current for next iteration
                tail_sum += f_logsig;
                // also store vecI in shared for later use if needed
                vecI_shared[i] = i_val;
            }
            if (USE_DBG) {
                for (uint i = 0; i < L; ++i) {
                    dbg[i] = vecA_shared[i];
                    dbg[L + i] = vecFlogsig_masked_shared[i];
                    dbg[2 * L + i] = vecI_shared[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // STEP 3: Compute scaG and scaAmax (single-thread, then broadcast)
        threadgroup float sca_shared[2]; // [0] = scaG_final, [1] = scaMinter_next_val
        if (tx == 0 && ty == 0) {
            float scaG_sum = 0.0f;
            float scaAmax_max = -INFINITY;
            for (uint i = 0; i < L; ++i) {
                // reconstruct logsig F directly from tail sums: we already computed via loop
                // but simplest is to read F and compute logsig again here
                uint f_idx = idx_b_BNH * str_vecFI_B_NH + k * L + i;
                float f_val = vecF[f_idx];
                scaG_sum += -log(1.0f + exp(-f_val));
                scaAmax_max = fmax(scaAmax_max, vecA_shared[i]);
            }
            
            if (USE_DBG) { dbg[3 * L] = scaG_sum; }

            float scaMinter_next = fmax(scaG_sum + scaMinter_k_val_shared[0], scaAmax_max);
            sca_shared[0] = scaG_sum;
            sca_shared[1] = scaMinter_next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scaG_final = sca_shared[0];
        float scaMinter_next_val = sca_shared[1];
        float scaMinter_curr = scaMinter_k_val_shared[0];

        // Compute normalized gates
        float scaGbar_k_val = exp(scaG_final + scaMinter_curr - scaMinter_next_val);

        // Load matK and matV tiles cooperatively into threadgroup memory
        threadgroup float matK_tile[16][64];  // (siz_b_DHQK, L)
        threadgroup float matV_tile[64][16];  // (L, siz_b_DHHV)

        // Load K tile (transposed): each thread loads one element
        for (uint krow = ty; krow < siz_b_DHQK; krow += siz_b_DHQK) {
            for (uint kcol = tx; kcol < L; kcol += siz_b_DHHV) {
                uint k_global_row = idx_b_DHQK * siz_b_DHQK + krow;
                uint k_global_col = k * L + kcol;
                float k_val = 0.0f;
                if (k_global_row < DHQK && k_global_col < S) {
                    uint k_idx = idx_b_BNH * str_matK_B_NH
                               + k_global_row * str_matK_DHQK
                               + k_global_col * str_matK_S;
                    k_val = matK[k_idx];
                }
                if (krow < siz_b_DHQK && kcol < L) {
                    matK_tile[krow][kcol] = k_val;
                }
            }
        }

        // Load V tile: each thread loads one element
        for (uint vrow = ty; vrow < L; vrow += siz_b_DHQK) {
            for (uint vcol = tx; vcol < siz_b_DHHV; vcol += siz_b_DHHV) {
                uint v_global_row = k * L + vrow;
                uint v_global_col = idx_b_DHHV * siz_b_DHHV + vcol;
                float v_val = 0.0f;
                if (v_global_row < S && v_global_col < DHHV) {
                    uint v_idx = idx_b_BNH * str_matV_B_NH
                               + v_global_row * str_matV_S
                               + v_global_col * str_matV_DHHV;
                    v_val = matV[v_idx];
                }
                if (vrow < L && vcol < siz_b_DHHV) {
                    matV_tile[vrow][vcol] = v_val;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute matKbar_k_val = matK_k_val * vecAbar (element-wise multiply by column)
        // vecAbar_k_val = exp(vecA_k_val - scaMinter_next_val)
        // CRITICAL: Use vecA_shared (threadgroup memory), not vecA_local!
        threadgroup float matKbar_tile[16][64];
        for (uint krow = ty; krow < siz_b_DHQK; krow += siz_b_DHQK) {
            for (uint kcol = tx; kcol < L; kcol += siz_b_DHHV) {
                float vecAbar_val = exp(vecA_shared[kcol] - scaMinter_next_val);
                if (krow < siz_b_DHQK && kcol < L) {
                    matKbar_tile[krow][kcol] = matK_tile[krow][kcol] * vecAbar_val;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update matC: matC_k_val = scaGbar * matC_k_val + matKbar @ matV
        // Compute matmul: matKbar (siz_b_DHQK, L) @ matV (L, siz_b_DHHV)
        if (tx < siz_b_DHHV && ty < siz_b_DHQK) {
            float acc = 0.0f;
            for (uint p = 0; p < L; ++p) {
                acc = fma(matKbar_tile[ty][p], matV_tile[p][tx], acc);
            }
            matC_k_val[ty][tx] = scaGbar_k_val * matC_k_val[ty][tx] + acc;
        }

        // Update vecN: vecN_k_val = scaGbar * vecN_k_val + sum(matKbar, axis=1)
        if (tx == 0 && ty < siz_b_DHQK) {
            float sum_row = 0.0f;
            for (uint p = 0; p < L; ++p) {
                sum_row += matKbar_tile[ty][p];
            }
            vecN_k_val[ty] = scaGbar_k_val * vecN_k_val[ty] + sum_row;
        }

        // Update scaMinter_k_val
        if (tx == 0 && ty == 0) {
            scaMinter_k_val_shared[0] = scaMinter_next_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store final states (k = NC)
    uint k_final = NC;
    if (k_final % save_states_every_nth_chunk == 0) {
        uint idx_k_save = k_final / save_states_every_nth_chunk;

        // Store matC_k_val tile
        uint matC_row = idx_b_DHQK * siz_b_DHQK + ty;
        uint matC_col = idx_b_DHHV * siz_b_DHHV + tx;
        if (tx < siz_b_DHHV && ty < siz_b_DHQK && matC_row < DHQK && matC_col < DHHV) {
            uint idx = idx_b_BNH * str_matCstates_B_NH
                     + idx_k_save * DHQK * DHHV
                     + matC_row * str_matCstates_NCDHQK
                     + matC_col * str_matCstates_DHHV;
            matC_states[idx] = matC_k_val[ty][tx];
        }

        // Store vecN_k_val
        if (idx_b_DHHV == 0 && tx == 0 && ty < siz_b_DHQK) {
            uint vecN_idx = idx_b_DHQK * siz_b_DHQK + ty;
            if (vecN_idx < DHQK) {
                uint idx = idx_b_BNH * str_vecNstates_B_NH
                         + idx_k_save * DHQK
                         + vecN_idx * str_vecNstates_NCDHQK;
                vecN_states[idx] = vecN_k_val[ty];
            }
        }

        // Store scaMinter_k_val
        if (idx_b_DHQK == 0 && idx_b_DHHV == 0 && tx == 0 && ty == 0) {
            uint idx = idx_b_BNH * str_scaMinterstates_B_NH + idx_k_save;
            scaMinter_states[idx] = scaMinter_k_val_shared[0];
        }
    }
"""

# Register kernel compiler (lazy compilation on first use)
def _compile_recurrent_kernel():
    """Compiler function - called once on first kernel access."""
    return mx.fast.metal_kernel(name="mlstm_recurrent_fw_C",
                                input_names=["matK", "matV", "vecF", "vecI", "matC_initial", "vecN_initial",
                                             "scaMinter_initial", "params", "strides"],
                                output_names=["matC_states", "vecN_states", "scaMinter_states", "dbg"], header=HEADER,
                                source=RECURRENT_FW_C_SRC)

# Registry removed - kernels are called directly by NCPS cells
# from xlstm_metal.blocks.mlstm.kern.mlx_jit.kernel_registry import register_kernel
# register_kernel('fw_recurrent', _compile_recurrent_kernel)

def _get_kernel():
    """Get compiled kernel (compiles on first call)."""
    return _compile_recurrent_kernel()

def mlstm_chunkwise_recurrent_fw_C_metal(
    matK: mx.array,  # (B, NH, S, DHQK)
    matV: mx.array,  # (B, NH, S, DHHV)
    vecF: mx.array,  # (B, NH, S)
    vecI: mx.array,  # (B, NH, S)
    matC_initial: Optional[mx.array],  # (B, NH, DHQK, DHHV)
    vecN_initial: Optional[mx.array],  # (B, NH, DHQK)
    scaMinter_initial: Optional[mx.array],  # (B, NH)
    NC: int,
    L: int,
    siz_b_DHQK: int = 16,
    siz_b_DHHV: int = 16,
    save_states_every_nth_chunk: int = 1,
    dbg: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array] | Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Metal kernel for recurrent forward computation of mLSTM chunk states.

    Returns:
        matC_states: (B, NH, (NC+1)*DHQK, DHHV)
        vecN_states: (B, NH, (NC+1)*DHQK)
        scaMinter_states: (B, NH, NC+1)
        dbg_out: (L * 3 + 1,) debug buffer
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[3]

    # Validate input dimensions
    assert matK.shape == (B, NH, S, DHQK), f"matK shape mismatch: {matK.shape} vs ({B}, {NH}, {S}, {DHQK})"
    assert matV.shape == (B, NH, S, DHHV), f"matV shape mismatch: {matV.shape} vs ({B}, {NH}, {S}, {DHHV})"
    assert vecF.shape == (B, NH, S), f"vecF shape mismatch: {vecF.shape} vs ({B}, {NH}, {S})"
    assert vecI.shape == (B, NH, S), f"vecI shape mismatch: {vecI.shape} vs ({B}, {NH}, {S})"

    # Prepare parameter buffer
    USE_INITIAL_STATE = 1 if matC_initial is not None else 0
    USE_DBG = 1 if dbg is not None else 0
    params = mx.array([B, NH, S, DHQK, DHHV, NC, L, siz_b_DHQK, siz_b_DHHV,
                       save_states_every_nth_chunk, USE_INITIAL_STATE, USE_DBG], dtype=mx.uint32)

    # Prepare strides buffer (all as if row-contiguous)
    # For K: (B, NH, S, DHQK) -> strides (NH*S*DHQK, S*DHQK, DHQK, 1)
    # For V: (B, NH, S, DHHV) -> strides (NH*S*DHHV, S*DHHV, DHHV, 1)
    strides = mx.array([
        NH * S * DHQK,  # str_matK_B_NH
        DHQK,           # str_matK_S
        1,              # str_matK_DHQK
        NH * S * DHHV,  # str_matV_B_NH
        DHHV,           # str_matV_S
        1,              # str_matV_DHHV
        NH * S,         # str_vecFI_B_NH
        (NC + 1) * DHQK * DHHV,  # str_matCstates_B_NH
        DHHV,           # str_matCstates_NCDHQK
        1,              # str_matCstates_DHHV
        (NC + 1) * DHQK,  # str_vecNstates_B_NH
        1,              # str_vecNstates_NCDHQK
        NC + 1,         # str_scaMinterstates_B_NH
        1,              # str_scaMinterstates_NC
        NH * DHQK * DHHV,  # str_matCinitial_B_NH
        DHHV,           # str_matCinitial_DHQK
        1,              # str_matCinitial_DHHV
        NH * DHQK,      # str_vecNinitial_B_NH
        1,              # str_vecNinitial_DHQK
        NH,             # str_scaMinterinitial_B_NH
    ], dtype=mx.uint32)

    # Allocate output states
    matC_states = mx.zeros((B, NH, (NC + 1) * DHQK, DHHV), dtype=matK.dtype)
    vecN_states = mx.zeros((B, NH, (NC + 1) * DHQK), dtype=matK.dtype)
    scaMinter_states = mx.zeros((B, NH, NC + 1), dtype=matK.dtype)

    # Default initial states if not provided
    if matC_initial is None:
        matC_initial = mx.zeros((B, NH, DHQK, DHHV), dtype=matK.dtype)
    if vecN_initial is None:
        vecN_initial = mx.zeros((B, NH, DHQK), dtype=matK.dtype)
    if scaMinter_initial is None:
        scaMinter_initial = mx.zeros((B, NH), dtype=matK.dtype)
    # Track whether caller asked for debug output; allocate buffer only if requested
    dbg_requested = dbg is not None
    if not dbg_requested:
        # Allocate a temporary buffer to satisfy kernel signature
        dbg = mx.zeros((L * 3 + 1,))

    # Launch pre-compiled kernel: grid over (DHQK/siz_b_DHQK, DHHV/siz_b_DHHV, B*NH)
    num_tiles_DHQK = (DHQK + siz_b_DHQK - 1) // siz_b_DHQK
    num_tiles_DHHV = (DHHV + siz_b_DHHV - 1) // siz_b_DHHV
    grid = (num_tiles_DHQK, num_tiles_DHHV, B * NH)
    threadgroup = (siz_b_DHHV, siz_b_DHQK, 1)

    outputs = _get_kernel()(
        inputs=[matK, matV, vecF, vecI, matC_initial, vecN_initial,
                scaMinter_initial, params, strides],
        output_shapes=[matC_states.shape, vecN_states.shape, scaMinter_states.shape, (L * 3 + 1,)],
        output_dtypes=[matK.dtype, matK.dtype, matK.dtype, dbg.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )

    matC_states, vecN_states, scaMinter_states, dbg_out = outputs

    # Preserve backward compatibility: only return dbg if explicitly requested
    if dbg_requested:
        return matC_states, vecN_states, scaMinter_states, dbg_out
    else:
        return matC_states, vecN_states, scaMinter_states


__all__ = ['mlstm_chunkwise_recurrent_fw_C_metal']
