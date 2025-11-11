"""sLSTM Output Cell - handles output processing.

This is the "after" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The output cell transforms kernel outputs back to input space:
- Per-head group normalization (MultiHeadLayerNorm)
- Final linear projection

It contains NO recurrence logic.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.mlstm.multihead_norm.multihead_norm import (
    MultiHeadLayerNorm
)


class sLSTMOutputCell(nn.Module):
    """
    sLSTM Output Cell - output transformation only.

    Handles all output processing for sLSTM:
    - Per-head group normalization (MultiHeadLayerNorm)
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

        # Multi-head group normalization (per-head)
        # Canonical uses MultiHeadLayerNorm (not RMS) with force_float32=True
        self.group_norm = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim, eps=eps)

        # Final output projection
        self.out_proj = nn.Linear(hidden_size, input_size, bias=use_bias)

    def __call__(
            self,
            h: mx.array,
    ) -> mx.array:
        """
        Process kernel output to final output.

        Args:
            h: Hidden states from kernel [B, S, NH, H]

        Returns:
            output: Final output [B, S, input_size]
        """
        B, S, NH, H = h.shape

        # Apply group norm (expects [B, S, NH, H], returns flattened [B, S, NH*H])
        h_norm = self.group_norm(h)

        # Output projection
        output = self.out_proj(h_norm)  # [B, S, input_size]

        return output


__all__ = ['sLSTMOutputCell']
