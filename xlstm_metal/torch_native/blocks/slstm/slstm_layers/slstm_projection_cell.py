"""sLSTM Projection Cell - handles all input projections.

This is the "before" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The projection cell transforms raw inputs into gate pre-activations
and z (cell input candidate). It contains NO recurrence logic.

Based on canonical xLSTM sLSTMLayer structure where:
- i, f gates use conv'd input (if conv enabled)
- z, o gates use raw input
- All gates get soft-capped
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_conv1d.causal_conv1d import CausalConv1d


class sLSTMProjectionCell(nn.Module):
    """
    sLSTM Projection Cell - input transformation only.

    Handles all input projections for sLSTM:
    - Optional causal Conv1d with SiLU activation
    - Gate projections: i, f (from conv'd), z, o (from raw)
    - Soft capping applied to gate pre-activations

    No recurrence, no output processing - just projections.

    Args:
        input_size: Input dimension (embedding_dim)
        num_heads: Number of sLSTM heads
        head_dim: Hidden dimension per head
        conv1d_kernel_size: Conv kernel size (0 = disabled, default 4)
        use_bias: Whether to use bias in z projection
        gate_soft_cap: Soft cap value for gates (default 15.0)
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            head_dim: int,
            conv1d_kernel_size: int = 4,
            use_bias: bool = False,
            gate_soft_cap: float = 15.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv1d_kernel_size = conv1d_kernel_size
        self.gate_soft_cap = gate_soft_cap

        hidden_size = num_heads * head_dim

        # Optional depthwise causal Conv1d
        if conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=conv1d_kernel_size,
                groups=input_size,  # depthwise
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv1d = None

        # Gate projections
        self.igate_proj = nn.Linear(input_size, num_heads)
        self.fgate_proj = nn.Linear(input_size, num_heads)
        self.ogate_proj = nn.Linear(input_size, num_heads)

        # z projection (cell input candidate)
        self.z_proj = nn.Linear(input_size, hidden_size, bias=use_bias)

    def soft_cap(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft capping: cap * tanh(x / cap)."""
        if self.gate_soft_cap is None:
            return x
        cap = torch.tensor(self.gate_soft_cap, dtype=x.dtype, device=x.device)
        return cap * torch.tanh(x / cap)

    def forward(
            self,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project inputs to gate pre-activations and z.

        Args:
            x: Input [B, S, input_size]

        Returns:
            z: Cell input candidate [B, S, NH, H]
            i_preact: Input gate pre-activation [B, S, NH] (soft-capped)
            f_preact: Forget gate pre-activation [B, S, NH] (soft-capped)
            o_preact: Output gate pre-activation [B, S, NH] (soft-capped)
            x_conv: Conv'd input (or raw if no conv) [B, S, input_size]
        """
        B, S, D = x.shape

        if self.conv1d is not None:
            # Conv1d expects [B, D, S]
            x_permuted = x.permute(0, 2, 1)
            x_conv_permuted = self.conv_act(self.conv1d(x_permuted))
            x_conv = x_conv_permuted.permute(0, 2, 1)  # Back to [B, S, D]
        else:
            x_conv = x

        # Project gates
        i_preact = self.soft_cap(self.igate_proj(x_conv))
        f_preact = self.soft_cap(self.fgate_proj(x_conv))
        o_preact = self.soft_cap(self.ogate_proj(x))

        # z projection from raw input
        z = self.z_proj(x)

        # Reshape z to multi-head: [B, S, NH*H] -> [B, S, NH, H]
        z = z.view(B, S, self.num_heads, self.head_dim)

        return z, i_preact, f_preact, o_preact, x_conv


__all__ = ['sLSTMProjectionCell']
