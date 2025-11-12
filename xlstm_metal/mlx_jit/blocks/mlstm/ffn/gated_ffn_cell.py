"""MLX implementation of Gated FFN as an NCPS cell.

This cell implements SwiGLU-style gated feed-forward network
compatible with the NCPS wiring framework.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class GatedFFNCell(nn.Module):
    """
    Gated Feed-Forward Network cell for NCPS.
    
    Implements SwiGLU gating: gate(x) * up(x) -> down
    
    This is a "neuron" in NCPS terminology but encapsulates
    the full FFN transformation.
    
    Args:
        input_size: Input dimension (embedding_dim)
        hidden_size: Intermediate dimension (proj_up_dim)
        activation: Activation function name ('silu', 'gelu', 'relu')
        use_bias: Whether to use bias in linear layers
        dropout: Optional dropout rate
        sparsity_mask: Optional sparsity mask for connections
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            activation: str = "silu",
            use_bias: bool = False,
            dropout: Optional[float] = None,
            sparsity_mask: Optional[mx.array] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        # Separate gate and value projections
        self.proj_up_gate = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.proj_up = nn.Linear(input_size, hidden_size, bias=use_bias)

        # Down projection
        self.proj_down = nn.Linear(hidden_size, input_size, bias=use_bias)

        # Activation
        if activation == "silu":
            self.act_fn = nn.silu
        elif activation == "gelu":
            self.act_fn = nn.gelu
        elif activation == "relu":
            self.act_fn = nn.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

        # Sparsity mask (for NCPS wiring)
        if sparsity_mask is not None:
            mask = mx.array(sparsity_mask) if not isinstance(sparsity_mask, mx.array) else sparsity_mask
            self._sparsity_mask = mx.abs(mask)
        else:
            self._sparsity_mask = None

    def __call__(self, x: mx.array, state: Optional[mx.array] = None) -> tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass.
        
        Args:
            x: Input [B, S, input_size]
            state: Unused (stateless cell), kept for NCPS compatibility
            
        Returns:
            (output, state): Output [B, S, input_size] and state (None for FFN)
        """
        # Separate gate and value projections
        gate = self.proj_up_gate(x)  # [B, S, hidden_size]
        z = self.proj_up(x)  # [B, S, hidden_size]

        # Gating: act(gate) * z
        gated = mx.multiply(self.act_fn(gate), z)

        # Apply sparsity mask if present
        if self._sparsity_mask is not None:
            gated = mx.multiply(gated, self._sparsity_mask)

        # Project down
        y = self.proj_down(gated)  # [B, S, input_size]

        # Apply dropout
        if self.dropout is not None:
            y = self.dropout(y)

        # Return output and state (None for stateless FFN)
        return y, None

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "use_bias": self.use_bias,
            "dropout": self.dropout.p if self.dropout else None,
        }
