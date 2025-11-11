"""mLSTM Recurrent Kernel Cell - pure sequential recurrence.

This is the "during" cell in the mLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The recurrent kernel cell implements step-by-step sequential recurrence.
Suitable for inference and autoregressive generation.

It contains ONLY recurrence logic, no projections or output processing.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class mLSTMRecurrentKernelCell(nn.Module):
    """
    mLSTM Recurrent Kernel Cell - sequential recurrence only.

    Implements step-by-step sequential mLSTM recurrence.
    Processes one timestep at a time - suitable for inference.

    No projections, no output processing - pure recurrence.

    Args:
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head
        v_dim_per_head: Value dimension per head
        eps: Numerical stability epsilon
    """

    def __init__(
            self,
            num_heads: int,
            qk_dim_per_head: int,
            v_dim_per_head: int,
            eps: float = 1e-6,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.eps = eps

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
        Apply sequential mLSTM recurrence.

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
        B, NH, S, DH_qk = q.shape
        DH_v = v.shape[-1]

        # Initialize state
        if state is None:
            C = mx.zeros((B, NH, DH_qk, DH_v))
            n = mx.zeros((B, NH, DH_qk))
            m = mx.zeros((B, NH))
        else:
            C, n, m = state
            C = C.astype(mx.float32)
            n = n.astype(mx.float32)
            m = m.astype(mx.float32)

        # Sequential processing
        h_list = []

        for t in range(S):
            # Extract timestep
            q_t = q[:, :, t, :]  # [B, NH, DH_qk]
            k_t = k[:, :, t, :]  # [B, NH, DH_qk]
            v_t = v[:, :, t, :]  # [B, NH, DH_v]
            i_t = i_preact[:, :, t]  # [B, NH]
            f_t = f_preact[:, :, t]  # [B, NH]

            # Stabilized exponential gates
            # m_t = max(f_log + m_{t-1}, i_t)
            one = mx.array(1.0, dtype=f_t.dtype)
            f_log = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_t)))))
            m_new = mx.maximum(mx.add(f_log, m), i_t)

            # Normalized gates
            f_gate = mx.exp(mx.subtract(mx.add(f_log, m), m_new))  # [B, NH]
            i_gate = mx.exp(mx.subtract(i_t, m_new))  # [B, NH]

            # Update state
            # C_t = f_t * C_{t-1} + i_t * (k_t ⊗ v_t)
            kv_outer = k_t[:, :, :, None] * v_t[:, :, None, :]  # [B, NH, DH_qk, DH_v]
            C = f_gate[:, :, None, None] * C + i_gate[:, :, None, None] * kv_outer

            # n_t = f_t * n_{t-1} + i_t * k_t
            n = f_gate[:, :, None] * n + i_gate[:, :, None] * k_t

            # Update m
            m = m_new

            # Compute output: h_t = (C_t @ q_t) / (|q·n| + exp(-m) + eps)
            q_scaled = q_t * mx.rsqrt(mx.array(self.qk_dim_per_head, dtype=q_t.dtype))

            # Numerator
            h_num = mx.sum(C * q_scaled[:, :, :, None], axis=2)  # [B, NH, DH_v]

            # Denominator
            qn_dot = mx.sum(n * q_scaled, axis=2, keepdims=True)  # [B, NH, 1]
            max_val = mx.exp(mx.negative(m))[:, :, None]
            h_den = mx.maximum(mx.abs(qn_dot), max_val) + self.eps

            h_t = h_num / h_den  # [B, NH, DH_v]
            h_list.append(h_t)

        # Stack outputs
        h = mx.stack(h_list, axis=2)  # [B, NH, S, DH_v]

        # Final state
        new_state = (C, n, m)

        return h, new_state


__all__ = ['mLSTMRecurrentKernelCell']
