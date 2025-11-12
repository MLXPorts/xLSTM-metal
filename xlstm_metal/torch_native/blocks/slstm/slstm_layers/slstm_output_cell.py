"""sLSTM Output Cell - handles output processing.

This is the "after" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The output cell transforms kernel outputs back to input space:
- Per-head group normalization
- Final linear projection

It contains NO recurrence logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class sLSTMOutputCell(nn.Module):
    """
    sLSTM Output Cell - output transformation only.

    Handles all output processing for sLSTM:
    - Per-head group normalization
    - Final projection back to input space

    No recurrence - just output transformations.

    Args:
        input_size: Input dimension (for final output)
        num_heads: Number of sLSTM heads
        head_dim: Hidden dimension per head
        use_bias: Whether to use bias in output projection
        eps: Epsilon for normalization
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            head_dim: int,
            use_bias: bool = False,
            eps: float = 1e-6,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        hidden_size = num_heads * head_dim

        # Per-head group normalization
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_size, eps=eps)

        # Final output projection
        self.out_proj = nn.Linear(hidden_size, input_size, bias=use_bias)

    def forward(
            self,
            h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process kernel output to final output.

        Args:
            h: Hidden states from kernel [B, S, NH, H]

        Returns:
            output: Final output [B, S, input_size]
        """
        B, S, NH, H = h.shape
        
        # Reshape for GroupNorm: [B*S, NH*H]
        h_reshaped = h.view(B * S, NH * H)
        
        # GroupNorm expects [N, C], so we permute to put channels first
        # h_reshaped is [B*S, NH*H], we need to think of it as [B*S, C]
        # where C = NH*H. GroupNorm will see num_groups=NH, so each group is H channels.
        # This seems to match the intention.
        h_norm = self.group_norm(h_reshaped)
        
        # Reshape back to [B, S, NH*H]
        h_norm = h_norm.view(B, S, NH * H)

        # Output projection
        output = self.out_proj(h_norm)  # [B, S, input_size]

        return output


__all__ = ['sLSTMOutputCell']
