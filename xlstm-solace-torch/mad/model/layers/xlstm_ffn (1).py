#!/usr/bin/env python
# Copyright (c) NXAI GmbH and its affiliates 2024
# Sydney Bach, Solace Harmony

"""
Feed-Forward Network block as MAD-style layer.

Implements gated FFN (SwiGLU-style) for xLSTM channel mixing.
"""

import torch
import torch.nn as nn

from xlstm_solace_torch.mad.init import small_init_init_, wang_init_


class GatedFFN(nn.Module):
    """Gated Feed-Forward Network (SwiGLU style).

    Args:
        dim: Model dimension
        proj_factor: Expansion factor (default 2.667 to match canonical)
        round_up_to: Round inner dim to multiple of this (default 64)
        bias: Use bias in linear layers (default False)
        dropout: Dropout rate (default 0.0)
        act: Activation function (default SiLU)
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        dim: int,
        proj_factor: float = 2.667,
        round_up_to: int = 64,
        bias: bool = False,
        dropout: float = 0.0,
        act: nn.Module = None,
        num_blocks: int = 1,
        *args,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        # Calculate inner dimension and round up
        inner_dim = int(dim * proj_factor)
        inner_dim = round_up_to * ((inner_dim + round_up_to - 1) // round_up_to)

        self.inner_dim = inner_dim

        # Gated projection: two paths
        self.proj_up_gate = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_up = nn.Linear(dim, inner_dim, bias=bias)

        # Down projection
        self.proj_down = nn.Linear(inner_dim, dim, bias=bias)

        # Activation
        self.act = act if act is not None else nn.SiLU()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, S, D]

        Returns:
            Output tensor [B, S, D]
        """
        # Gated path
        gate = self.act(self.proj_up_gate(x))
        up = self.proj_up(x)

        # Element-wise multiplication (gating)
        gated = gate * up

        # Project down
        y = self.dropout(self.proj_down(gated))

        return y

    def reset_parameters(self):
        """Initialize weights using canonical xLSTM initialization.

        - proj_up_gate and proj_up: small_init (for input projections)
        - proj_down: wang_init (for output projection, scaled by num_blocks)

        This matches the canonical xLSTM implementation for numerical parity.
        """
        # Input projections use small_init
        small_init_init_(self.proj_up_gate.weight, dim=self.dim)
        small_init_init_(self.proj_up.weight, dim=self.dim)

        # Output projection uses wang_init (scaled by num_blocks for residual connections)
        wang_init_(self.proj_down.weight, dim=self.inner_dim, num_blocks=self.num_blocks)

        # Biases (if present) are zero-initialized by PyTorch default
        # No need to explicitly reinitialize them
