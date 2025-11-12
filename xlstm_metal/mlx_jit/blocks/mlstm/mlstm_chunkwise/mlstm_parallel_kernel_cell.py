"""mLSTM Parallel Kernel Cell – MLX Implementation (Chunkwise Recurrence)

Overview
--------
The parallel kernel cell is the **"during"** (recurrence) component of
the mLSTM pipeline using a **chunkwise parallel** strategy. It processes
sequences in fixed-size chunks, computing:
  1. **Inter-chunk** recurrence (sequential across chunks)
  2. **Intra-chunk** attention (parallel within each chunk)

This approach balances parallelism (for throughput) with recurrent memory
(for long-range dependencies).

Two-Phase Algorithm
-------------------
Phase 1 (Sequential across chunks):
  For each chunk k = 0..NC-1:
    Compute inter-chunk states (C_k, n_k, m_k) from prior state + chunk content.
  This produces NC + 1 boundary states (including initial state).

Phase 2 (Parallel within chunks):
  For all positions i in all chunks (fully parallel):
    Compute hidden state h_i using:
      - Intra-chunk attention (causal within chunk)
      - Inter-chunk contribution from boundary state C_{k-1}
  Uses Metal-optimized kernel for high throughput.

Why Chunkwise?
--------------
- Full parallel attention over long sequences: O(S²) memory + compute
- Full recurrent: O(S) memory but sequential (slow for long S)
- Chunkwise: O(S · L) intra-chunk + O(S/L) inter-chunk (practical tradeoff)

Chunk Size L
------------
Typical values: 64, 128. Larger L increases intra-chunk parallelism but
requires more memory. Smaller L reduces per-chunk cost but increases
sequential overhead from inter-chunk updates.

State Structure
---------------
State tuple = (C, n, m):
  C : [B, NH, DH_qk, DH_v]  matrix-memory (outer product accumulator)
  n : [B, NH, DH_qk]        normalizer vector
  m : [B, NH]               scalar stabilizer (log-space gating)

Padding
-------
If sequence length S is not divisible by chunk_size L, the inputs are
zero-padded to NC * L (NC = ceil(S / L)). Output is then unpadded back
to length S.

Metal Kernels
-------------
Both recurrent (inter-chunk) and parallel (intra-chunk) phases call
Metal-optimized kernels for efficient execution on Apple Silicon.

Numerical Stability
-------------------
- Input/forget gate preactivations are processed in log-space to avoid
  exponential overflow.
- Query scaling (1 / sqrt(DH_qk)) applied before attention-like operations.
- Mixed precision: compute in `compute_dtype`, store state in `state_dtype`.

Parity
------
Logic mirrors torch-native `mLSTMParallelKernelCell` for cross-backend testing.
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
    """Chunkwise parallel recurrence kernel for mLSTM (no projections, pure memory).

    Parameters
    ----------
    num_heads : int
        Number of attention heads (NH).
    qk_dim_per_head : int
        Query/key dimension per head.
    v_dim_per_head : int
        Value dimension per head.
    chunk_size : int, default 64
        Chunk length L for parallel processing.
    eps : float, default 1e-6
        Numerical stability epsilon.
    compute_dtype : mx.Dtype, default mx.float32
        Dtype for forward pass activations.
    state_dtype : mx.Dtype, default mx.float32
        Dtype for recurrent state storage (C, n, m).

    Returns (forward)
    -----------------
    h : mx.array [B, NH, S, DH_v]
        Hidden states for all timesteps.
    new_state : (C, n, m)
        Updated boundary state for next call.
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
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:  # noqa: D401
        """Execute two-phase chunkwise parallel mLSTM recurrence.

        Parameters
        ----------
        q : mx.array [B, NH, S, DH_qk]
            Query projections.
        k : mx.array [B, NH, S, DH_qk]
            Key projections.
        v : mx.array [B, NH, S, DH_v]
            Value projections.
        i_preact : mx.array [B, NH, S]
            Input gate preactivations.
        f_preact : mx.array [B, NH, S]
            Forget gate preactivations.
        state : tuple | None
            Previous recurrent state (C, n, m) or None for initialization.

        Returns
        -------
        h : mx.array [B, NH, S, DH_v]
            Hidden states computed via chunkwise algorithm.
        new_state : (C, n, m)
            Final recurrent state (boundary of last chunk).
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
