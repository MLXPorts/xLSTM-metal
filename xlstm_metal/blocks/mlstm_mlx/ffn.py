#!/usr/bin/env python
"""
Gated Feed-Forward Network (FFN) for xLSTM

Implements the SwiGLU-style gated FFN used in xLSTM-7B.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Literal


@dataclass
class FFNConfig:
    """
    Configuration for Gated FFN.

    From xlstm_7b_model:
        embedding_dim: 4096
        proj_factor: ~2.67 (proj_up_dim = 10944)
        act_fn: "swish" (SwiGLU variant)
        use_bias: False
    """
    embedding_dim: int = 4096
    proj_factor: float = 2.671875  # 10944 / 4096
    act_fn: Literal["gelu", "swish", "relu"] = "swish"
    use_bias: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        self.proj_up_dim = int(self.embedding_dim * self.proj_factor)


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network with SwiGLU-style gating.

    Architecture:
        x -> proj_up -> split -> [gate, up_proj]
        gate -> act_fn(gate)
        out = act_fn(gate) * up_proj
        out -> proj_down -> y

    Weight structure (xLSTM-7B):
        proj_up.weight: [2 * proj_up_dim, embedding_dim]  # [21888, 4096]
        proj_down.weight: [embedding_dim, proj_up_dim]     # [4096, 10944]

    Note: proj_up outputs 2x proj_up_dim to handle both gate and up_proj
    """

    def __init__(self, config: FFNConfig):
        super().__init__()
        self.config = config

        # Project up to 2x intermediate dim (for gate + up_proj)
        self.proj_up = nn.Linear(
            config.embedding_dim,
            2 * config.proj_up_dim,
            bias=config.use_bias
        )

        # Project down from intermediate to embedding dim
        self.proj_down = nn.Linear(
            config.proj_up_dim,
            config.embedding_dim,
            bias=config.use_bias
        )

        # Activation function
        if config.act_fn == "gelu":
            self.act_fn = nn.gelu
        elif config.act_fn == "swish":
            self.act_fn = nn.silu  # SiLU = Swish
        elif config.act_fn == "relu":
            self.act_fn = nn.relu
        else:
            raise ValueError(f"Unknown activation function: {config.act_fn}")

        # Dropout (usually 0 for inference)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass

        Args:
            x: Input tensor [B, S, embedding_dim]

        Returns:
            y: Output tensor [B, S, embedding_dim]
        """
        # Project up and split
        up = self.proj_up(x)  # [B, S, 2 * proj_up_dim]

        # Split into gate and up_proj
        gate_preact, up_proj = mx.split(up, 2, axis=-1)  # Each: [B, S, proj_up_dim]

        # Apply gating: act_fn(gate) * up_proj
        gated = self.act_fn(gate_preact) * up_proj  # [B, S, proj_up_dim]

        # Project down
        y = self.proj_down(gated)  # [B, S, embedding_dim]

        # Apply dropout if configured
        if self.dropout is not None:
            y = self.dropout(y)

        return y


class FFNBlock(nn.Module):
    """
    Complete FFN block with pre-normalization and residual.

    Matches the structure:
        x -> RMSNorm -> GatedFFN -> (+) residual
    """

    def __init__(self, config: FFNConfig, norm_eps: float = 1e-6):
        super().__init__()
        self.config = config

        # Import here to avoid circular dependency
        from .components import RMSNorm

        self.norm_ffn = RMSNorm(
            num_features=config.embedding_dim,
            eps=norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=True
        )

        self.ffn = GatedFFN(config)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with pre-normalization and residual

        Args:
            x: Input [B, S, embedding_dim]

        Returns:
            x_out: Output with residual [B, S, embedding_dim]
        """
        # Pre-normalization
        x_norm = self.norm_ffn(x)

        # FFN
        x_ffn = self.ffn(x_norm)

        # Residual connection
        x_out = x + x_ffn

        return x_out
