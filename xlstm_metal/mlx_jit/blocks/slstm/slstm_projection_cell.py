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
from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn


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

        # Optional Conv1d for temporal context (canonical applies to i, f gates)
        if conv1d_kernel_size > 0:
            self.causal_pad = conv1d_kernel_size - 1
            self.conv1d = nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=conv1d_kernel_size,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv1d = None

        # Gate projections
        # Canonical: i, f use conv'd input (if enabled)
        # z, o use raw input
        # All gates have bias=True in canonical (for proper initialization)
        self.igate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.ogate_proj = nn.Linear(input_size, num_heads, bias=True)

        # z projection (cell input candidate)
        self.z_proj = nn.Linear(input_size, hidden_size, bias=use_bias)

    def soft_cap(self, x: mx.array) -> mx.array:
        """Apply soft capping: cap * tanh(x / cap)."""
        if self.gate_soft_cap is None:
            return x
        cap = mx.array(self.gate_soft_cap, dtype=x.dtype)
        return mx.multiply(cap, mx.tanh(mx.divide(x, cap)))

    def __call__(
            self,
            x: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
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
        B, S, _ = x.shape

        # Apply Conv1d if enabled (canonical applies to i, f gates)
        if self.conv1d is not None:
            # Causal padding: [B, S, D] needs to be [B, D, S] for Conv1d
            x_transposed = x.transpose(0, 2, 1)  # [B, D, S]

            # Causal padding on sequence dimension
            if self.causal_pad > 0:
                padding = mx.zeros((B, self.input_size, self.causal_pad), dtype=x.dtype)
                x_transposed = mx.concatenate([padding, x_transposed], axis=2)

            # Apply conv and activation
            x_conv_transposed = self.conv_act(self.conv1d(x_transposed))  # [B, D, S]
            x_conv = x_conv_transposed.transpose(0, 2, 1)  # [B, S, D]
        else:
            x_conv = x

        # Project gates
        # CRITICAL: i, f use conv'd input; o uses raw input (canonical pattern)
        i_preact = self.soft_cap(self.igate_proj(x_conv))  # [B, S, NH]
        f_preact = self.soft_cap(self.fgate_proj(x_conv))  # [B, S, NH]
        o_preact = self.soft_cap(self.ogate_proj(x))  # [B, S, NH] - raw input

        # z projection from raw input
        z = self.z_proj(x)  # [B, S, NH*H]

        # Reshape z to multi-head: [B, S, NH*H] -> [B, S, NH, H]
        z = z.reshape(B, S, self.num_heads, self.head_dim)

        return z, i_preact, f_preact, o_preact, x_conv


__all__ = ['sLSTMProjectionCell']
