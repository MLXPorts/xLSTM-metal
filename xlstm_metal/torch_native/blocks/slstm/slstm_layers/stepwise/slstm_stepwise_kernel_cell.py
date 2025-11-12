"""sLSTM Stepwise Kernel Cell - pure PyTorch recurrence.

This is the "during" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The stepwise kernel cell implements sequential recurrence with canonical
sLSTM equations for numerical stability.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class sLSTMStepwiseKernelCell(nn.Module):
    """
    sLSTM Stepwise Kernel Cell - sequential recurrence only.

    Implements canonical sLSTM recurrence from xlstm package with:
    - Proper stability clamps
    - logsigmoid for forget gate
    - tanh(z) for cell input

    No projections, no output processing - pure recurrence.

    Args:
        num_heads: Number of sLSTM heads
        head_dim: Hidden dimension per head
        eps: Numerical stability epsilon
    """

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps

    def forward(
            self,
            z: torch.Tensor,
            i_preact: torch.Tensor,
            f_preact: torch.Tensor,
            o_preact: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Apply single-step sLSTM recurrence.

        Args:
            z: Cell input candidate [B, NH, H]
            i_preact: Input gate pre-activation [B, NH] (soft-capped)
            f_preact: Forget gate pre-activation [B, NH] (soft-capped)
            o_preact: Output gate pre-activation [B, NH] (soft-capped)
            state: Optional previous state (c, n, m)
                   c: [B, NH, H] - cell state
                   n: [B, NH, H] - normalizer
                   m: [B, NH] - stabilizer

        Returns:
            h: Hidden states [B, NH, H]
            new_state: Updated state (c, n, m)
        """
        B, NH, H = z.shape

        # Initialize state if needed
        if state is None:
            c_state = torch.zeros((B, NH, H), dtype=z.dtype, device=z.device)
            n_state = torch.zeros((B, NH, H), dtype=z.dtype, device=z.device)
            m_state = torch.zeros((B, NH), dtype=z.dtype, device=z.device)
        else:
            c_state, n_state, m_state = state

        # Stabilized gates
        m_new = torch.maximum(m_state + F.logsigmoid(f_preact), i_preact)
        
        i_gate = torch.exp(i_preact - m_new)
        f_gate = torch.exp(m_state + F.logsigmoid(f_preact) - m_new)
        o_gate = torch.sigmoid(o_preact)

        # Update cell state and normalizer
        c_new = f_gate.unsqueeze(-1) * c_state + i_gate.unsqueeze(-1) * torch.tanh(z)
        n_new = f_gate.unsqueeze(-1) * n_state + i_gate.unsqueeze(-1)

        # Compute hidden state
        h = o_gate.unsqueeze(-1) * c_new / (torch.abs(n_new) + self.eps)

        new_state = (c_new, n_new, m_new)
        return h, new_state


__all__ = ['sLSTMStepwiseKernelCell']
