"""
xLSTM Block for MAD Architecture

Combines mLSTM + FFN blocks matching canonical xLSTM structure:
    x -> xlstm_norm -> mLSTM -> (+) residual
    x -> ffn_norm -> FFN -> (+) residual
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .block import mLSTMLayer, mLSTMConfig
from .components import RMSNorm
from xlstm_metal.mlx_native.blocks.ffn import GatedFFN


@dataclass
class xLSTMBlockConfig:
    """
    Configuration for complete xLSTM block (mLSTM + FFN).

    Matches canonical xLSTMBlockConfig from xlstm package.
    """
    # mLSTM configuration
    embedding_dim: int = 4096
    num_heads: int = 8
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0

    # FFN configuration
    ffn_proj_factor: float = 2.671875  # 10944 / 4096
    ffn_act_fn: str = "swish"

    # Shared configuration
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    dropout: float = 0.0

    # Inference settings
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True
    chunk_size: int = 64

    def __post_init__(self):
        # Create mLSTM config
        self.mlstm_config = mLSTMConfig(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            gate_soft_cap=self.gate_soft_cap,
            use_bias=self.use_bias,
            norm_eps=self.norm_eps,
            norm_reduction_force_float32=self.norm_reduction_force_float32,
            eps=self.eps,
            inference_state_dtype=self.inference_state_dtype,
            return_last_states=self.return_last_states,
            chunk_size=self.chunk_size
        )

        # FFN config is not needed - we pass parameters directly to GatedFFN


class xLSTMBlock(nn.Module):
    """
    Complete xLSTM block matching canonical structure.

    Structure:
        x -> xlstm_norm -> mLSTMLayer -> (+) residual
        x -> ffn_norm -> FFN -> (+) residual

    This matches the canonical xLSTMBlock from xlstm package:
        - Pre-normalization (RMSNorm for xLSTM-7B)
        - Residual connections
        - Optional FFN (always included for xLSTM-7B)

    Note: Canonical uses LayerNorm, xLSTM-7B uses RMSNorm.
    """

    def __init__(self, config: xLSTMBlockConfig):
        super().__init__()
        self.config = config

        # Pre-normalization for mLSTM
        self.xlstm_norm = RMSNorm(num_features=config.embedding_dim, eps=config.norm_eps, use_bias=config.use_bias,
                                  force_float32_reductions=config.norm_reduction_force_float32)

        # mLSTM layer (without its own norm - we handle that here)
        self.xlstm = mLSTMLayer(config.mlstm_config)

        # Pre-normalization for FFN
        self.ffn_norm = RMSNorm(num_features=config.embedding_dim, eps=config.norm_eps, use_bias=config.use_bias,
                                force_float32_reductions=config.norm_reduction_force_float32)

        # FFN (GatedFFN - no separate norm inside)
        self.ffn = GatedFFN(
            hidden_size=config.embedding_dim,
            ffn_proj_factor=config.ffn_proj_factor,
            use_bias=config.use_bias
        )

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass matching canonical xLSTMBlock.

        Args:
            x: Input [B, S, embedding_dim]
            state: Optional mLSTM state (c, n, m)

        Returns:
            x_out: Output [B, S, embedding_dim]
            state: Updated mLSTM state or None
        """
        # mLSTM block: x + mLSTM(norm(x))
        x_norm = self.xlstm_norm(x)
        x_mlstm, new_state = self.xlstm(x_norm, state)
        x = mx.add(x, x_mlstm)

        # FFN block: x + FFN(norm(x))
        x_norm = self.ffn_norm(x)
        x_ffn = self.ffn(x_norm)
        x = mx.add(x, x_ffn)

        return x, new_state
