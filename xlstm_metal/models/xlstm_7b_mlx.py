#!/usr/bin/env python
"""
xLSTM-7B MAD Model for MLX

Standalone implementation matching canonical xlstm.xlstm_large.model structure.
Uses MAD blocks with MLX backend.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from ..blocks.mlstm_mlx.xlstm_block import xLSTMBlock, xLSTMBlockConfig
from ..blocks.mlstm_mlx.components import RMSNorm, soft_cap

# Type aliases matching canonical
mLSTMLayerStateType = Tuple[mx.array, mx.array, mx.array]  # (c, n, m)
mLSTMStateType = Dict[int, Optional[mLSTMLayerStateType]]


@dataclass
class xLSTM7BConfig:
    """
    Configuration for xLSTM-7B model matching canonical structure.

    Canonical equivalent: xlstm.xlstm_large.model.xLSTMLargeConfig
    """
    # Model architecture
    embedding_dim: int = 4096
    num_heads: int = 8
    num_blocks: int = 32
    vocab_size: int = 50304

    # mLSTM layer
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0

    # FFN
    ffn_proj_factor: float = 2.671875  # 10944 / 4096
    ffn_act_fn: str = "swish"

    # Normalization
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    add_out_norm: bool = True

    # Inference
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True

    # Capping
    output_logit_soft_cap: float = 30.0


class xLSTM7B(nn.Module):
    """
    xLSTM-7B model for MLX.

    Canonical equivalent: xlstm.xlstm_large.model.xLSTMLarge

    Structure:
        embedding -> backbone (32 blocks) -> lm_head
    """

    config_class = xLSTM7BConfig

    def __init__(self, config: xLSTM7BConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Block stack
        self.backbone = xLSTM7BBlockStack(config)

        # LM head (no bias)
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

    def load_weights(self, npz_path: str):
        """
        Load pretrained weights from NPZ file.

        Args:
            npz_path: Path to xlstm_7b_mlx_converted.npz

        Uses mad.utils.weight_loader to map NPZ weights to MAD blocks.
        """
        from ..utils.weight_loader import load_xLSTM_7b_weights

        # Load weights into blocks
        embedding_weight, head_weight = load_xLSTM_7b_weights(
            npz_path,
            self.backbone.blocks
        )

        # Set embedding and head weights
        self.embedding.weight = embedding_weight
        self.lm_head.weight = head_weight

        print(f"✅ Loaded pretrained weights from {npz_path}")

    def __call__(
        self,
        x: mx.array,
        state: Optional[mLSTMStateType] = None
    ) -> Tuple[mx.array, Optional[mLSTMStateType]]:
        """
        Forward pass.

        Args:
            x: Input token IDs [B, S]
            state: Optional state dict {block_idx: (c, n, m)}

        Returns:
            logits: Output logits [B, S, vocab_size]
            state: Updated state dict (if return_last_states=True)
        """
        assert x.ndim == 2, f"Input must be [B, S], got {x.shape}"

        # Embed tokens
        x = self.embedding(x)  # [B, S, embedding_dim]

        # Process through backbone
        x, state = self.backbone(x, state)  # [B, S, embedding_dim]

        # LM head
        logits = self.lm_head(x)  # [B, S, vocab_size]

        # Apply output soft-cap
        logits_capped = soft_cap(logits, self.config.output_logit_soft_cap)

        if self.config.return_last_states:
            return logits_capped, state
        else:
            return logits_capped, None


class xLSTM7BBlockStack(nn.Module):
    """
    Stack of xLSTM blocks with final normalization.

    Canonical equivalent: xlstm.xlstm_large.model.xLSTMLargeBlockStack
    """

    config_class = xLSTM7BConfig

    def __init__(self, config: xLSTM7BConfig):
        super().__init__()
        self.config = config

        # Create block config
        block_config = xLSTMBlockConfig(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            qk_dim_factor=config.qk_dim_factor,
            v_dim_factor=config.v_dim_factor,
            gate_soft_cap=config.gate_soft_cap,
            ffn_proj_factor=config.ffn_proj_factor,
            ffn_act_fn=config.ffn_act_fn,
            use_bias=config.use_bias,
            norm_eps=config.norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            eps=config.eps,
            inference_state_dtype=config.inference_state_dtype,
            return_last_states=config.return_last_states
        )

        # Create blocks
        self.blocks = [xLSTMBlock(block_config) for _ in range(config.num_blocks)]

        # Final normalization
        if config.add_out_norm:
            self.out_norm = RMSNorm(
                num_features=config.embedding_dim,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32
            )
        else:
            self.out_norm = lambda x: x  # Identity

    def __call__(
        self,
        x: mx.array,
        state: Optional[mLSTMStateType] = None
    ) -> Tuple[mx.array, mLSTMStateType]:
        """
        Forward through all blocks.

        Args:
            x: Input [B, S, embedding_dim]
            state: Optional state dict {block_idx: (c, n, m)}

        Returns:
            x: Output [B, S, embedding_dim]
            state: Updated state dict
        """
        # Initialize state if None
        if state is None:
            state = {i: None for i in range(len(self.blocks))}

        # Process through each block
        for i, block in enumerate(self.blocks):
            block_state = state[i]
            x, block_state_new = block(x, block_state)

            # Update state
            state[i] = block_state_new

        # Final normalization
        x = self.out_norm(x)

        return x, state
