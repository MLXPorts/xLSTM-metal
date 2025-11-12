"""mLSTM Parallel Kernel Cell - pure parallel/chunkwise recurrence.

This is the "during" cell in the mLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The parallel kernel cell implements chunkwise parallel recurrence
using Metal-accelerated kernels. It contains ONLY recurrence logic,
no projections or output processing.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .forward import (
    mlstm_chunkwise_recurrent_fw_C_metal,
    mlstm_chunkwise_parallel_fw_Hintra_metal
)


class mLSTMParallelKernelCell(nn.Module):
    """
    mLSTM Parallel Kernel Cell - chunkwise parallel recurrence only.

    Implements two-phase chunkwise parallel algorithm:
    1. Recurrent phase: Compute inter-chunk states sequentially
    2. Parallel phase: Compute intra-chunk outputs in parallel

    No projections, no output processing - pure recurrence.

    Args:
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head
        v_dim_per_head: Value dimension per head
        chunk_size: Chunk size for parallel processing
        eps: Numerical stability epsilon
    """

    def __init__(
            self,
            num_heads: int,
            qk_dim_per_head: int,
            v_dim_per_head: int,
            chunk_size: int = 64,
            eps: float = 1e-6,
            compute_dtype: mx.Dtype = mx.float32,
            state_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.chunk_size = chunk_size
        self.eps = eps
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype
        qk_dim_value = mx.array(qk_dim_per_head, dtype=self.compute_dtype)
        self.qk_scale = mx.divide(mx.array(1.0, dtype=self.compute_dtype), mx.sqrt(qk_dim_value))

    def __call__(
            self,
            q: mx.array,
            k: mx.array,
            v: mx.array,
            i_preact: mx.array,
            f_preact: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:
        """
        Apply chunkwise parallel mLSTM recurrence.

        Args:
            q: Query [B, NH, S, DH_qk]
            k: Key [B, NH, S, DH_qk]
            v: Value [B, NH, S, DH_v]
            i_preact: Input gate pre-activation [B, NH, S]
            f_preact: Forget gate pre-activation [B, NH, S]
            state: Optional previous state (C, n, m)
                   C: [B, NH, DH_qk, DH_v]
                   n: [B, NH, DH_qk]
                   m: [B, NH]

        Returns:
            h: Hidden states [B, NH, S, DH_v]
            new_state: Updated state (C, n, m)
        """
        q = mx.array(q, dtype=self.compute_dtype)
        k = mx.array(k, dtype=self.compute_dtype)
        v = mx.array(v, dtype=self.compute_dtype)
        i_preact = mx.array(i_preact, dtype=self.compute_dtype)
        f_preact = mx.array(f_preact, dtype=self.compute_dtype)

        B, NH, S, _ = q.shape

        # Extract or initialize state
        c_initial = mx.array(state[0], dtype=self.state_dtype) if state and state[0] is not None else None
        n_initial = mx.array(state[1], dtype=self.state_dtype) if state and state[1] is not None else None
        m_initial = mx.array(state[2], dtype=self.state_dtype) if state and state[2] is not None else None

        # Pad sequence to multiple of chunk_size
        NC = (S + self.chunk_size - 1) // self.chunk_size
        L = self.chunk_size

        if S % L != 0:
            pad_len = NC * L - S
            q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            i_preact = mx.pad(i_preact, [(0, 0), (0, 0), (0, pad_len)])
            f_preact = mx.pad(f_preact, [(0, 0), (0, 0), (0, pad_len)])
            S_padded = NC * L
        else:
            S_padded = S

        # ===== Phase 1: Recurrent (Inter-chunk states) =====
        matC_states, vecN_states, scaMinter_states = (
            mlstm_chunkwise_recurrent_fw_C_metal(
                matK=k,
                matV=v,
                vecF=f_preact,
                vecI=i_preact,
                matC_initial=c_initial,
                vecN_initial=n_initial,
                scaMinter_initial=m_initial,
                NC=NC,
                L=L,
                state_dtype=self.state_dtype,
                )
        )

        # ===== Phase 2: Parallel (Intra-chunk outputs) =====
        # Reshape for chunked processing
        vecI_chunked = i_preact.reshape(B, NH, NC, L)
        vecF_chunked = f_preact.reshape(B, NH, NC, L)

        # Compute cumulative log forget gates (for intra-chunk)
        one = mx.array(1.0, dtype=vecF_chunked.dtype)
        vecF_logsig = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(vecF_chunked)))))
        vecB = mx.cumsum(vecF_logsig, axis=-1)

        # Query scaling factor
        qk_scale = mx.array(self.qk_scale, dtype=self.compute_dtype)

        # Parallel kernel (intra-chunk)
        matHout, vecNout, vecMout = mlstm_chunkwise_parallel_fw_Hintra_metal(matQ=q,
                                                                             matK=k,
                                                                             matV=v,
                                                                             matC_states=matC_states,
                                                                             vecN_states=vecN_states,
                                                                             scaMinter_states=scaMinter_states,
                                                                             vecI=vecI_chunked,
                                                                             vecB=vecB, NC=NC, L=L,
                                                                             qk_scale=qk_scale, eps=self.eps)

        # Unpad if necessary
        if S != S_padded:
            matHout = matHout[:, :, :S, :]

        # Cast activations back to compute dtype for downstream layers
        if matHout.dtype != self.compute_dtype:
            matHout = mx.array(matHout, dtype=self.compute_dtype)

        # Extract final state (last chunk)
        qk_dim = self.qk_dim_per_head
        new_state = (
            matC_states[:, :, -qk_dim:, :],
            vecN_states[:, :, -qk_dim:],
            scaMinter_states[:, :, -1],
        )

        return matHout, new_state


__all__ = ['mLSTMParallelKernelCell']
