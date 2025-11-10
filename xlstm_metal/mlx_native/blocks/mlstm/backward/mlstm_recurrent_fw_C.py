"""
mLSTM Chunkwise MLX Implementation (High-Level Operations)

NOT a Metal kernel - this uses high-level MLX array operations.
Transliterated from mlstm_kernels.triton.chunkwise.xl_chunk Triton kernels.

This is pure MLX using mx.matmul, mx.exp, mx.maximum, etc.
Metal kernel implementation: see mlstm_metal/ (COMPLETE - 6/6 kernels ported).

Algorithm (from Triton mlstm_chunkwise_parallel_fw_Hintra.py and mlstm_chunkwise_recurrent_fw_C.py):
1. Phase 1 (Recurrent): Compute inter-chunk states C_k, n_k, m_k sequentially
   - vecB = cumsum(logsigmoid(f)) - backward gate contribution
   - vecA = vecB[-1] - vecB + vecI - forward gate contribution
   - scaG = sum(logsigmoid(f)) - chunk transition scalar
   - Update: C_k = scaGbar * C_{k-1} + K^T @ (Abar ⊙ V)

2. Phase 2 (Parallel): Compute outputs H within each chunk in parallel
   - Intra-chunk: attention within chunk using causal mask
   - Inter-chunk: contribution from previous state C_{k-1}
   - Combine: H = (H_inter + ratio * H_intra) / denom

Critical details from canonical mLSTM:
- Forget gate uses logsigmoid before exponential
- Query is scaled by 1/√d_qk
- Denominator uses max(|q·n|, exp(-m)) + eps
- C state shape is [B, NH, QK_DH, V_DH] (k⊗v not v⊗k)
"""

from __future__ import annotations

from typing import Tuple, Optional

import mlx.core as mx


def _mlstm_chunkwise_recurrent_fw_C_native(
        k: mx.array,  # (B, NH, S, DHQK)
        v: mx.array,  # (B, NH, S, DHHV)
        vec_b: mx.array,  # (B, NH, NC, L) - cumsum(logsigmoid(f))
        vec_i: mx.array,  # (B, NH, NC, L) - input gate preact
        c_initial: Optional[mx.array] = None,  # (B, NH, DHQK, DHHV)
        n_initial: Optional[mx.array] = None,  # (B, NH, DHQK)
        m_initial: Optional[mx.array] = None,  # (B, NH)
        chunk_size: int = 64,
        num_chunks: int = 1,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Recurrent computation of inter-chunk states (Triton mlstm_chunkwise_recurrent_fw_C.py).

    Processes chunks sequentially to compute C_k, n_k, m_k states.

    Returns:
        matC_states: (B, NH, (NC+1) * DHQK, DHHV) - all chunk states
        vecN_states: (B, NH, (NC+1) * DHQK) - all normalizer states
        scaM_states: (B, NH, (NC+1)) - all max states
    """
    B, NH, S, DHQK = k.shape
    DHHV = v.shape[3]
    NC = num_chunks
    L = chunk_size

    # Initialize running states
    c_k = mx.zeros((B, NH, DHQK, DHHV), dtype=k.dtype) if c_initial is None else c_initial
    n_k = mx.zeros((B, NH, DHQK), dtype=k.dtype) if n_initial is None else n_initial
    m_k = mx.zeros((B, NH), dtype=k.dtype) if m_initial is None else m_initial

    # Compute vecA and scaG from vecB and vecI (Triton lines 91-93, 166-172)
    vec_a = mx.add(mx.subtract(vec_b[:, :, :, -1:], vec_b), vec_i)  # (B, NH, NC, L)
    sca_g = vec_b[:, :, :, -1]  # (B, NH, NC)
    sca_a_max = mx.max(vec_a, axis=-1)  # (B, NH, NC)

    # Store states at each chunk
    c_states_list = [c_k]
    n_states_list = [n_k]
    m_states_list = [m_k]

    # Iterate over chunks (Triton lines 94-198)
    for chunk_idx in range(NC):
        # Update m_inter (Triton lines 175-176)
        sca_a_max_k = sca_a_max[:, :, chunk_idx]
        sca_g_k = sca_g[:, :, chunk_idx]
        m_k_next = mx.maximum(mx.add(sca_g_k, m_k), sca_a_max_k)

        # Load chunk data
        k_chunk = k[:, :, chunk_idx * L:(chunk_idx + 1) * L, :]  # (B, NH, L, DHQK)
        v_chunk = v[:, :, chunk_idx * L:(chunk_idx + 1) * L, :]  # (B, NH, L, DHHV)
        vec_a_k = vec_a[:, :, chunk_idx, :]  # (B, NH, L)

        # Compute normalized gates (Triton lines 183-184)
        vec_abar_k = mx.exp(mx.subtract(vec_a_k, m_k_next[:, :, None]))  # (B, NH, L)
        sca_gbar_k = mx.exp(mx.subtract(mx.add(sca_g_k, m_k), m_k_next))  # (B, NH)

        # Update C state: C_k = scaGbar * C_{k-1} + K^T @ (Abar ⊙ V) (Triton lines 187-191)
        # k_chunk: (B, NH, L, DHQK), vec_abar_k: (B, NH, L)
        k_gated = mx.multiply(k_chunk, vec_abar_k[:, :, :, None])  # (B, NH, L, DHQK)
        # k_gated^T @ v_chunk: (B, NH, DHQK, L) @ (B, NH, L, DHHV) = (B, NH, DHQK, DHHV)
        c_k = mx.add(
            mx.multiply(sca_gbar_k[:, :, None, None], c_k),
            mx.matmul(k_gated.transpose(0, 1, 3, 2), v_chunk)
        )

        # Update n state: n_k = scaGbar * n_{k-1} + sum(K_gated, axis=2) (Triton lines 194-195)
        n_k = mx.add(mx.multiply(sca_gbar_k[:, :, None], n_k), mx.sum(k_gated, axis=2))

        # Move to next iteration
        m_k = m_k_next

        c_states_list.append(c_k)
        n_states_list.append(n_k)
        m_states_list.append(m_k)

    # Concatenate all states
    matC_states = mx.concatenate(c_states_list, axis=2)  # (B, NH, (NC+1)*DHQK, DHHV)
    vecN_states = mx.concatenate(n_states_list, axis=2)  # (B, NH, (NC+1)*DHQK)
    scaM_states = mx.stack(m_states_list, axis=2)  # (B, NH, NC+1)

    return matC_states, vecN_states, scaM_states


def mlstm_chunkwise_recurrent_fw_C(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        i_preact: mx.array,
        f_preact: mx.array,
        chunk_size: int = 64,
        c_initial: Optional[mx.array] = None,
        n_initial: Optional[mx.array] = None,
        m_initial: Optional[mx.array] = None,
        eps: float = 1e-6,
        return_last_states: bool = True
) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
    """
    Chunkwise mLSTM using MLX operations (based on Triton kernels).

    Implements the algorithm from mlstm_kernels.triton.chunkwise.xl_chunk.
    Two-phase processing:
    1. Recurrent: Sequential computation of inter-chunk states C_k, n_k, m_k
    2. Parallel: Parallel computation of outputs within each chunk

    Args:
        q: Query tensor [B, NH, S, QK_DH]
        k: Key tensor [B, NH, S, QK_DH]
        v: Value tensor [B, NH, S, V_DH]
        i_preact: Input gate pre-activation [B, NH, S]
        f_preact: Forget gate pre-activation [B, NH, S]
        chunk_size: Size of chunks (default 64)
        c_initial: Initial covariance [B, NH, QK_DH, V_DH] or None
        n_initial: Initial normalizer [B, NH, QK_DH] or None
        m_initial: Initial running max [B, NH] or None
        eps: Numerical stability constant
        return_last_states: Whether to return final states

    Returns:
        h: Hidden states [B, NH, S, V_DH]
        state: Tuple (c_final, n_final, m_final) if return_last_states else None
    """
    B, NH, S, QK_DH = q.shape
    V_DH = v.shape[3]

    assert S % chunk_size == 0, f"Sequence length {S} must be divisible by chunk_size {chunk_size}"
    NC = S // chunk_size
    qk_scale = mx.rsqrt(mx.array(QK_DH, dtype=q.dtype))

    # Reshape to chunks: (B, NH, S) -> (B, NH, NC, L)
    i_preact = i_preact.reshape(B, NH, NC, chunk_size)
    f_preact = f_preact.reshape(B, NH, NC, chunk_size)

    # Compute vecB = cumsum(logsigmoid(f)) (Triton line 157, 313)
    one = mx.array(1.0, dtype=f_preact.dtype)
    f_logsig = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_preact)))))  # logsigmoid
    vec_b = mx.cumsum(f_logsig, axis=-1)  # (B, NH, NC, L)

    # Phase 1: Recurrent computation of inter-chunk states
    matC_states, vecN_states, scaM_states = mlstm_chunkwise_recurrent_fw_C(
        k=k,
        v=v,
        vec_b=vec_b,
        vec_i=i_preact,
        c_initial=c_initial,
        n_initial=n_initial,
        m_initial=m_initial,
        chunk_size=chunk_size,
        num_chunks=NC,
    )

    # Phase 2: Parallel computation of outputs (Triton mlstm_chunkwise_parallel_fw_Hintra.py lines 85-271)
    # Reshape Q, K, V to chunks
    q_chunks = q.reshape(B, NH, NC, chunk_size, QK_DH)  # (B, NH, NC, L, DHQK)
    k_chunks = k.reshape(B, NH, NC, chunk_size, QK_DH)  # (B, NH, NC, L, DHQK)
    v_chunks = v.reshape(B, NH, NC, chunk_size, V_DH)  # (B, NH, NC, L, DHHV)

    # Use states up to but not including current chunk (Triton line 337-339)
    # matC_states shape: (B, NH, (NC+1)*DHQK, DHHV) -> take first NC chunks
    matC_k_states = matC_states[:, :, :NC * QK_DH, :].reshape(B, NH, NC, QK_DH, V_DH)
    vecN_k_states = vecN_states[:, :, :NC * QK_DH].reshape(B, NH, NC, QK_DH)
    scaM_k_states = scaM_states[:, :, :NC]  # (B, NH, NC)

    # Compute intra-chunk contribution (Triton lines 85-174)
    # matG = Q @ K^T (Triton lines 96-122)
    matG_chunk = mx.matmul(q_chunks, k_chunks.transpose(0, 1, 2, 4, 3))  # (B, NH, NC, L, L)

    # matDtilde = vecB[:, None] - vecB[None, :] + vecI[None, :] (Triton line 133)
    # vec_b, vec_i shape: (B, NH, NC, L)
    matF_logsig_chunk = mx.subtract(vec_b[:, :, :, :, None], vec_b[:, :, :, None, :])  # (B, NH, NC, L, L)
    matLogD_chunk = mx.add(matF_logsig_chunk, i_preact[:, :, :, None, :])  # (B, NH, NC, L, L)

    # Apply causal mask (Triton lines 137-140)
    # Create lower triangular mask
    causal_mask = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_))
    neg_inf = mx.array(-3.4028235e38, dtype=matLogD_chunk.dtype)
    matLogD_chunk = mx.where(causal_mask, matLogD_chunk, neg_inf)

    # Compute vecM_intra = max(matLogD_chunk, axis=-1) (Triton lines 143-144)
    vecM_intra = mx.max(matLogD_chunk, axis=-1)  # (B, NH, NC, L)
    min_max_val = mx.array(-10.0, dtype=vecM_intra.dtype)
    vecM_intra = mx.maximum(vecM_intra, min_max_val)  # MINIMUM_MAX_VAL

    # Compute vecM_combine (Triton lines 177-182)
    vecM_b_inter = mx.add(vec_b, scaM_k_states[:, :, :, None])  # (B, NH, NC, L)
    vecM_combine = mx.maximum(vecM_b_inter, vecM_intra)  # (B, NH, NC, L)

    # Compute matD = exp(matLogD - vecM_combine) (Triton line 150)
    matD_chunk = mx.exp(mx.subtract(matLogD_chunk, vecM_combine[:, :, :, :, None]))  # (B, NH, NC, L, L)

    # Compute matS = matG * qk_scale * matD (Triton line 153)
    matS_chunk = mx.multiply(mx.multiply(matG_chunk, qk_scale), matD_chunk)  # (B, NH, NC, L, L)

    # Compute H_intra = matS @ V (Triton lines 169-171)
    # This is accumulated in the Triton kernel, but we compute it directly here
    matH_intra = mx.matmul(matS_chunk, v_chunks)  # (B, NH, NC, L, DHHV)

    # Compute vecN_intra = sum(matS, axis=-1) (Triton line 156)
    vecN_intra = mx.sum(matS_chunk, axis=-1)  # (B, NH, NC, L)

    # Compute inter-chunk contribution (Triton lines 176-233)
    # vecBbar = exp(vecB + scaM_inter - vecM_combine) (Triton line 184)
    vecBbar = mx.exp(mx.subtract(vecM_b_inter, vecM_combine))  # (B, NH, NC, L)

    # matQbar = Q * vecBbar * qk_scale (Triton line 221)
    matQbar = mx.multiply(mx.multiply(q_chunks, vecBbar[:, :, :, :, None]), qk_scale)  # (B, NH, NC, L, DHQK)

    # matH_inter = matQbar @ matC_{k-1} (Triton line 227)
    matH_inter = mx.matmul(matQbar, matC_k_states)  # (B, NH, NC, L, DHHV)

    # vecN_inter = matQbar @ vecN_{k-1} (Triton line 233)
    vecN_inter = mx.sum(mx.multiply(matQbar, vecN_k_states[:, :, :, None, :]), axis=-1)  # (B, NH, NC, L)

    # Combine intra and inter contributions (Triton lines 235-250)
    # vecM_comb_ratio = exp(vecM_intra - vecM_combine) (Triton line 238)
    vecM_comb_ratio = mx.exp(mx.subtract(vecM_intra, vecM_combine))  # (B, NH, NC, L)

    # matH_comb_num = matH_inter + vecM_comb_ratio * matH_intra (Triton line 241)
    matH_comb_num = mx.add(matH_inter,
                           mx.multiply(vecM_comb_ratio[:, :, :, :, None], matH_intra))  # (B, NH, NC, L, DHHV)

    # vecN_comb_denom = max(|vecN_inter + vecM_comb_ratio * vecN_intra|, exp(-vecM_combine)) (Triton lines 244-247)
    vecN_comb = mx.add(vecN_inter, mx.multiply(vecM_comb_ratio, vecN_intra))  # (B, NH, NC, L)
    vecN_comb_denom = mx.maximum(mx.abs(vecN_comb), mx.exp(mx.negative(vecM_combine)))  # (B, NH, NC, L)

    # matH_out = matH_comb_num / (vecN_comb_denom + eps) (Triton line 250)
    eps_a = mx.array(eps, dtype=vecN_comb_denom.dtype)
    matH_out = mx.divide(matH_comb_num, mx.add(vecN_comb_denom[:, :, :, :, None], eps_a))  # (B, NH, NC, L, DHHV)

    # Reshape output back to (B, NH, S, DHHV)
    h = matH_out.reshape(B, NH, S, V_DH)

    if return_last_states:
        # Return final states (last in the states list)
        c_final = matC_states[:, :, -QK_DH:, :]
        n_final = vecN_states[:, :, -QK_DH:]
        m_final = scaM_states[:, :, -1]
        return h, (c_final, n_final, m_final)
    else:
        return h, None


__all__ = [ 'mlstm_chunkwise_recurrent_fw_C']