"""
mLSTM Chunkwise MLX Implementation - Parallel Phase (High-Level Operations)

NOT a Metal kernel - this uses high-level MLX array operations.
Transliterated from mlstm_kernels.triton.chunkwise.xl_chunk Triton kernels.

This implements Phase 2: Parallel computation of outputs within each chunk.
"""

from __future__ import annotations

from typing import Tuple, Optional

import mlx.core as mx

def mlstm_chunkwise_parallel_fw_C(
        q: mx.array,  # (B, NH, S, QK_DH)
        k: mx.array,  # (B, NH, S, QK_DH)
        v: mx.array,  # (B, NH, S, V_DH)
        i_preact: mx.array,  # (B, NH, S)
        f_preact: mx.array,  # (B, NH, S)
        matC_states: mx.array,  # (B, NH, (NC+1)*QK_DH, V_DH)
        vecN_states: mx.array,  # (B, NH, (NC+1)*QK_DH)
        scaM_states: mx.array,  # (B, NH, NC+1)
        chunk_size: int = 64,
        eps: float = 1e-6,
) -> mx.array:
    """
    Parallel computation of outputs within each chunk (Triton mlstm_chunkwise_parallel_fw_Hintra.py).

    Args:
        q: Query tensor [B, NH, S, QK_DH]
        k: Key tensor [B, NH, S, QK_DH]
        v: Value tensor [B, NH, S, V_DH]
        i_preact: Input gate pre-activation [B, NH, S]
        f_preact: Forget gate pre-activation [B, NH, S]
        matC_states: Inter-chunk C states from recurrent phase
        vecN_states: Inter-chunk N states from recurrent phase
        scaM_states: Inter-chunk M states from recurrent phase
        chunk_size: Size of chunks
        eps: Numerical stability constant

    Returns:
        h: Hidden states [B, NH, S, V_DH]
    """
    B, NH, S, QK_DH = q.shape
    V_DH = v.shape[3]
    NC = S // chunk_size
    qk_scale = mx.rsqrt(mx.array(QK_DH, dtype=q.dtype))

    # Reshape to chunks
    q_chunks = q.reshape(B, NH, NC, chunk_size, QK_DH)
    k_chunks = k.reshape(B, NH, NC, chunk_size, QK_DH)
    v_chunks = v.reshape(B, NH, NC, chunk_size, V_DH)
    i_preact_chunks = i_preact.reshape(B, NH, NC, chunk_size)
    f_preact_chunks = f_preact.reshape(B, NH, NC, chunk_size)

    # Compute vecB = cumsum(logsigmoid(f))
    one = mx.array(1.0, dtype=f_preact_chunks.dtype)
    f_logsig = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_preact_chunks)))))
    vec_b = mx.cumsum(f_logsig, axis=-1)

    # Use states up to but not including current chunk
    matC_k_states = matC_states[:, :, :NC * QK_DH, :].reshape(B, NH, NC, QK_DH, V_DH)
    vecN_k_states = vecN_states[:, :, :NC * QK_DH].reshape(B, NH, NC, QK_DH)
    scaM_k_states = scaM_states[:, :, :NC]

    # Intra-chunk contribution
    matG_chunk = mx.matmul(q_chunks, k_chunks.transpose(0, 1, 2, 4, 3))
    matF_logsig_chunk = mx.subtract(vec_b[:, :, :, :, None], vec_b[:, :, :, None, :])
    matLogD_chunk = mx.add(matF_logsig_chunk, i_preact_chunks[:, :, :, None, :])
    causal_mask = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_))
    neg_inf = mx.array(-3.4028235e38, dtype=matLogD_chunk.dtype)
    matLogD_chunk = mx.where(causal_mask, matLogD_chunk, neg_inf)
    vecM_intra = mx.max(matLogD_chunk, axis=-1)
    min_max_val = mx.array(-10.0, dtype=vecM_intra.dtype)
    vecM_intra = mx.maximum(vecM_intra, min_max_val)
    vecM_b_inter = mx.add(vec_b, scaM_k_states[:, :, :, None])
    vecM_combine = mx.maximum(vecM_b_inter, vecM_intra)
    matD_chunk = mx.exp(mx.subtract(matLogD_chunk, vecM_combine[:, :, :, :, None]))
    matS_chunk = mx.multiply(mx.multiply(matG_chunk, qk_scale), matD_chunk)
    matH_intra = mx.matmul(matS_chunk, v_chunks)
    vecN_intra = mx.sum(matS_chunk, axis=-1)

    # Inter-chunk contribution
    vecBbar = mx.exp(mx.subtract(vecM_b_inter, vecM_combine))
    matQbar = mx.multiply(mx.multiply(q_chunks, vecBbar[:, :, :, :, None]), qk_scale)
    matH_inter = mx.matmul(matQbar, matC_k_states)
    vecN_inter = mx.sum(mx.multiply(matQbar, vecN_k_states[:, :, :, None, :]), axis=-1)

    # Combine contributions
    vecM_comb_ratio = mx.exp(mx.subtract(vecM_intra, vecM_combine))
    matH_comb_num = mx.add(matH_inter, mx.multiply(vecM_comb_ratio[:, :, :, :, None], matH_intra))
    vecN_comb = mx.add(vecN_inter, mx.multiply(vecM_comb_ratio, vecN_intra))
    vecN_comb_denom = mx.maximum(mx.abs(vecN_comb), mx.exp(mx.negative(vecM_combine)))
    eps_a = mx.array(eps, dtype=vecN_comb_denom.dtype)
    matH_out = mx.divide(matH_comb_num, mx.add(vecN_comb_denom[:, :, :, :, None], eps_a))

    # Reshape output
    h = matH_out.reshape(B, NH, S, V_DH)
    return h

__all__ = ['mlstm_chunkwise_parallel_fw_C']