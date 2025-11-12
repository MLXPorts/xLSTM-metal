"""mLSTM Recurrent Kernel Cell - pure sequential recurrence.

This is the "during" cell in the mLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The recurrent kernel cell implements step-by-step sequential recurrence.
Suitable for inference and autoregressive generation.

It contains ONLY recurrence logic, no projections or output processing.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


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
            compute_dtype: torch.dtype = torch.float32,
            state_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.eps = eps
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            i_preact: torch.Tensor,
            f_preact: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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

        q = q.to(dtype=self.compute_dtype)
        k = k.to(dtype=self.compute_dtype)
        v = v.to(dtype=self.compute_dtype)
        i_preact = i_preact.to(dtype=self.compute_dtype)
        f_preact = f_preact.to(dtype=self.compute_dtype)

        # Initialize state
        if state is None:
            C = torch.zeros((B, NH, DH_qk, DH_v), dtype=self.state_dtype, device=q.device)
            n = torch.zeros((B, NH, DH_qk), dtype=self.state_dtype, device=q.device)
            m = torch.zeros((B, NH), dtype=self.state_dtype, device=q.device)
        else:
            C, n, m = state
            C = C.to(dtype=self.state_dtype)
            n = n.to(dtype=self.state_dtype)
            m = m.to(dtype=self.state_dtype)

        # Sequential processing
        h_list = []

        eps_tensor = torch.tensor(self.eps, dtype=self.compute_dtype, device=q.device)
        qk_dim_per_head_tensor = torch.tensor(self.qk_dim_per_head, dtype=self.compute_dtype, device=q.device)

        for t in range(S):
            # Extract timestep
            q_t = q[:, :, t, :]  # [B, NH, DH_qk]
            k_t = k[:, :, t, :]  # [B, NH, DH_qk]
            v_t = v[:, :, t, :]  # [B, NH, DH_v]
            i_t = i_preact[:, :, t]  # [B, NH]
            f_t = f_preact[:, :, t]  # [B, NH]

            # Stabilized exponential gates
            # m_t = max(f_log + m_{t-1}, i_t)
            one = torch.tensor(1.0, dtype=f_t.dtype, device=f_t.device)
            f_log = torch.neg(torch.log(torch.add(one, torch.exp(torch.neg(f_t)))))
            m_new = torch.maximum(torch.add(f_log, m), i_t)

            # Normalized gates
            f_gate = torch.exp(torch.subtract(torch.add(f_log, m), m_new))  # [B, NH]
            i_gate = torch.exp(torch.subtract(i_t, m_new))  # [B, NH]

            # Update state
            # C_t = f_t * C_{t-1} + i_t * (k_t ⊗ v_t)
            kv_outer = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # [B, NH, DH_qk, DH_v]
            kv_outer_state = kv_outer.to(dtype=self.state_dtype)
            f_gate_state = f_gate.to(dtype=self.state_dtype)
            i_gate_state = i_gate.to(dtype=self.state_dtype)
            C = f_gate_state.unsqueeze(-1).unsqueeze(-1) * C + i_gate_state.unsqueeze(-1).unsqueeze(-1) * kv_outer_state

            # n_t = f_t * n_{t-1} + i_t * k_t
            k_state = k_t.to(dtype=self.state_dtype)
            n = f_gate_state.unsqueeze(-1) * n + i_gate_state.unsqueeze(-1) * k_state

            # Update m
            m = m_new.to(dtype=self.state_dtype)

            # Compute output: h_t = (C_t @ q_t) / (|q·n| + exp(-m) + eps)
            q_scaled = q_t * torch.rsqrt(qk_dim_per_head_tensor)

            # Numerator
            C_compute = C.to(dtype=self.compute_dtype)
            h_num = torch.sum(C_compute * q_scaled.unsqueeze(-1), dim=2)  # [B, NH, DH_v]

            # Denominator
            n_compute = n.to(dtype=self.compute_dtype)
            qn_dot = torch.sum(n_compute * q_scaled, dim=2, keepdim=True)  # [B, NH, 1]
            max_val = torch.exp(torch.neg(m.to(dtype=self.compute_dtype))).unsqueeze(-1)
            h_den = torch.maximum(torch.abs(qn_dot), max_val) + eps_tensor

            h_t = h_num / h_den  # [B, NH, DH_v]
            h_list.append(h_t)

        # Stack outputs
        h = torch.stack(h_list, dim=2)  # [B, NH, S, DH_v]

        # Final state
        new_state = (C, n, m)

        return h, new_state


__all__ = ['mLSTMRecurrentKernelCell']
