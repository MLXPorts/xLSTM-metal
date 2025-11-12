"""
sLSTM PyTorch Kernels - Numerically Stable Implementation

Following canonical xlstm package implementation with:
- logsigmoid for forget gate
- tanh(z) for cell input
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def slstm_step_pytorch(
        z: torch.Tensor,
        i_preact: torch.Tensor,
        f_preact: torch.Tensor,
        o_preact: torch.Tensor,
        c_state: torch.Tensor,
        n_state: torch.Tensor,
        m_state: torch.Tensor,
        eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single sLSTM step using PyTorch with canonical equations.

    Implements numerically stable sLSTM from xlstm package.

    Args:
        z: Cell input candidate [B, NH, H]
        i_preact: Input gate pre-activation [B, NH] (soft-capped)
        f_preact: Forget gate pre-activation [B, NH] (soft-capped)
        o_preact: Output gate pre-activation [B, NH] (soft-capped)
        c_state: Cell state [B, NH, H]
        n_state: Normalizer state [B, NH, H]
        m_state: Stabilizer [B, NH]
        eps: Numerical stability epsilon

    Returns:
        h_out: Hidden output [B, NH, H]
        c_state_out: Updated cell state [B, NH, H]
        n_state_out: Updated normalizer [B, NH, H]
        m_state_out: Updated stabilizer [B, NH]
    """
    # Stabilizer update
    m_new = torch.maximum(m_state + F.logsigmoid(f_preact), i_preact)

    # Gate computation
    i_gate = torch.exp(i_preact - m_new)
    f_gate = torch.exp(m_state + F.logsigmoid(f_preact) - m_new)
    o_gate = torch.sigmoid(o_preact)

    # Unsqueeze gates for broadcasting
    i_gate_u = i_gate.unsqueeze(-1)
    f_gate_u = f_gate.unsqueeze(-1)
    o_gate_u = o_gate.unsqueeze(-1)

    # New cell state and normalizer
    c_new = f_gate_u * c_state + i_gate_u * torch.tanh(z)
    n_new = f_gate_u * n_state + i_gate_u

    # Hidden state
    h_out = o_gate_u * c_new / (torch.abs(n_new) + eps)

    return h_out, c_new, n_new, m_new


__all__ = ['slstm_step_pytorch']
