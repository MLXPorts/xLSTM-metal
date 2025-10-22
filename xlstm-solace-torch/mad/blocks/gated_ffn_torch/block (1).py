#!/usr/bin/env python
# Copyright (c) NXAI GmbH and its affiliates 2024
# Sydney Bach, Solace Harmony
# Gated FFN block implementation for MAD framework

"""
Gated feed-forward network block for the MAD framework.

SwiGLU-style architecture:
1. Input [B, S, D]
2. Two parallel projections: D → inner_dim
3. One goes through activation (gate), other is linear (up)
4. Element-wise multiply: gate * up
5. Project down: inner_dim → D
6. Output [B, S, D]
"""

import torch
import torch.nn as nn

from xlstm_solace_torch.mad.init import small_init_init_, wang_init_


class GatedFFN(nn.Module):
    """Gated feed-forward network with SwiGLU-style activation.

    Args:
        dim: Model dimension
        proj_factor: Expansion factor (default 2.667)
        round_to: Round inner_dim to multiple of this (default 64)
        bias: Use bias in projections (default False)
        dropout: Dropout rate (default 0.0)
        activation: Activation function (default SiLU)
        num_blocks: Total blocks for weight init (default 1)
    """

    def __init__(
        self,
        dim: int,
        proj_factor: float = 2.667,
        round_to: int = 64,
        bias: bool = False,
        dropout: float = 0.0,
        activation: nn.Module = None,
        num_blocks: int = 1,
    ):
        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        # Calculate inner dimension and round up
        inner_dim = int(dim * proj_factor)
        inner_dim = round_to * ((inner_dim + round_to - 1) // round_to)
        self.inner_dim = inner_dim

        # Gated projection: two parallel paths
        self.proj_gate = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_up = nn.Linear(dim, inner_dim, bias=bias)

        # Down projection
        self.proj_down = nn.Linear(inner_dim, dim, bias=bias)

        # Activation
        self.activation = activation if activation is not None else nn.SiLU()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, S, D]

        Returns:
            Output [B, S, D]
        """
        # Gated path
        gate = self.activation(self.proj_gate(x))  # [B, S, inner_dim]
        up = self.proj_up(x)  # [B, S, inner_dim]

        # Element-wise gating
        gated = gate * up  # [B, S, inner_dim]

        # Project down
        y = self.proj_down(gated)  # [B, S, D]

        # Dropout
        if self.dropout is not None:
            y = self.dropout(y)

        return y

    def reset_parameters(self):
        """Initialize weights using canonical xLSTM initialization."""
        # Input projections: small_init
        small_init_init_(self.proj_gate.weight, dim=self.dim)
        small_init_init_(self.proj_up.weight, dim=self.dim)

        # Output projection: wang_init (scaled by num_blocks)
        wang_init_(self.proj_down.weight, dim=self.inner_dim, num_blocks=self.num_blocks)
