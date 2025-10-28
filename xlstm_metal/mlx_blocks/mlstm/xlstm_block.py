#!/usr/bin/env python
"""
xLSTM Block - Wrapper around xLSTM Cells

Following NCPS architecture:
- Block wraps a cell (mLSTM, sLSTM, etc.)
- Adds pre-normalization
- Adds residual connections
- Adds FFN path
- Block is the unit wired together in circuits
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from .mlstm_cell import mLSTMCell, mLSTMCellConfig
from .block import mLSTMLayer, mLSTMConfig
from .ffn import GatedFFN, FFNConfig
from .components import RMSNorm


@dataclass
class xLSTMBlockConfig:
    """
    Configuration for xLSTM block (wrapper around cell).

    This config specifies both:
    - Cell configuration (which cell type and its params)
    - Block configuration (norms, FFN, etc.)
    """
    # Cell configuration
    embedding_dim: int
    num_heads: int
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0

    # FFN configuration
    ffn_proj_factor: float = 2.671875
    ffn_act_fn: str = "swish"

    # Shared configuration
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True
    chunk_size: int = 64
    dropout: float = 0.0

    # Cell type (for future extensibility)
    cell_type: str = "mlstm"  # "mlstm", "slstm", "conv1d"

    def __post_init__(self):
        """Validate and create sub-configs."""
        # Validate required architectural parameters
        required_fields = ['embedding_dim', 'num_heads', 'qk_dim_factor',
                          'v_dim_factor', 'gate_soft_cap', 'ffn_proj_factor']
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"Architectural parameter '{field}' must be provided from checkpoint")

        # Create cell config
        self.cell_config = mLSTMCellConfig(
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

        # Create FFN config
        self.ffn_config = FFNConfig(
            embedding_dim=self.embedding_dim,
            proj_factor=self.ffn_proj_factor,
            act_fn=self.ffn_act_fn,
            use_bias=self.use_bias,
            dropout=self.dropout
        )

        # Legacy support - create mLSTM config for old code
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


class xLSTMBlock(nn.Module):
    """
    xLSTM Block - Wrapper around xLSTM Cell.

    Following NCPS cell/block pattern:
    - Block wraps a cell (the actual computation)
    - Adds pre-normalization layers
    - Adds residual connections
    - Adds FFN path

    Structure:
        x -> xlstm_norm -> Cell (mLSTM/sLSTM) -> (+) residual
        x -> ffn_norm -> FFN -> (+) residual

    The block is the unit that gets wired together in neural circuits.
    The cell is the core computational unit that can have internal wiring.

    Args:
        config: xLSTMBlockConfig specifying cell type and parameters
        cell_wiring: Optional wiring for cell's internal connectivity
    """

    def __init__(
        self,
        config: xLSTMBlockConfig,
        cell_wiring: Optional[Any] = None
    ):
        super().__init__()
        self.config = config

        # Pre-normalization for cell
        self.xlstm_norm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32
        )

        # Create cell based on type
        if config.cell_type == "mlstm":
            # Use new cell-based architecture
            self.xlstm = mLSTMCell(config.cell_config, wiring=cell_wiring)
        elif config.cell_type == "slstm":
            # TODO: Implement sLSTMCell
            raise NotImplementedError("sLSTM cell not yet implemented")
        elif config.cell_type == "conv1d":
            # TODO: Implement Conv1dCell
            raise NotImplementedError("Conv1d cell not yet implemented")
        else:
            raise ValueError(f"Unknown cell type: {config.cell_type}")

        # Pre-normalization for FFN
        self.ffn_norm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32
        )

        # FFN
        self.ffn = GatedFFN(config.ffn_config)

    def __call__(
        self,
        x: mx.array,
        state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass through xLSTM block.

        Args:
            x: Input [B, S, embedding_dim]
            state: Optional cell state (format depends on cell type)

        Returns:
            x_out: Output [B, S, embedding_dim]
            state: Updated cell state or None
        """
        # Cell path: x + Cell(norm(x))
        x_norm = self.xlstm_norm(x)
        x_cell, new_state = self.xlstm(x_norm, state)
        x = mx.add(x, x_cell)  # residual

        # FFN path: x + FFN(norm(x))
        x_norm = self.ffn_norm(x)
        x_ffn = self.ffn(x_norm)
        x = mx.add(x, x_ffn)  # residual

        return x, new_state
