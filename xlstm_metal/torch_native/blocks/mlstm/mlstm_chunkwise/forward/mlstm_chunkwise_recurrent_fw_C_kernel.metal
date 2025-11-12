#include <metal_stdlib>
using namespace metal;

kernel void mlstm_recurrent_fw_C(
    device const float* matK [[buffer(0)]],
    device const float* matV [[buffer(1)]],
    device const float* vecF [[buffer(2)]],
    device const float* vecI [[buffer(3)]],
    device const float* matC_initial [[buffer(4)]],
    device const float* vecN_initial [[buffer(5)]],
    device const float* scaMinter_initial [[buffer(6)]],
    device float* matC_states [[buffer(7)]],
    device float* vecN_states [[buffer(8)]],
    device float* scaMinter_states [[buffer(9)]],
    device float* dbg [[buffer(10)]],
    constant uint* params [[buffer(11)]],
    constant uint* strides [[buffer(12)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {

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

}
