"""xLSTM-7B Cell - NCPS-compatible wrapper for full xLSTM block.

This cell wraps a complete xLSTM block (mLSTM + FFN) with proper parameter
handling for model loading from safetensors and config.json.

Follows NCPS patterns for clean composability and wiring.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_chunkwise.mlstm_neuron import mLSTMNeuron
from xlstm_metal.mlx_jit.blocks.rms_norm import RMSNormCell
from xlstm_metal.mlx_jit.utils import resolve_dtype


class mLSTMBlock(nn.Module):
    """
    Complete mLSTM (xlSTM-7B block) (mLSTM + FFN) for NCPS.
    
    This represents one full xLSTM block from the model:
        input -> norm_mlstm -> mLSTM -> residual
              -> norm_ffn -> FFN -> residual -> output
    
    Parameters are loaded from:
    - config.json: Model hyperparameters
    - safetensors: Pretrained weights (backbone.blocks.{i}.*)
    
    Args:
        block_index: Block index for weight loading (0-31 for xLSTM-7B)
        embedding_dim: Model dimension (default: 4096)
        num_heads: Number of attention heads (default: 8)
        qk_dim_factor: QK dimension factor (default: 0.5)
        v_dim_factor: V dimension factor (default: 1.0)
        gate_soft_cap: Gate soft cap value (default: 15.0)
        ffn_proj_factor: FFN projection factor (default: 2.667)
        ffn_round_up_to_multiple_of: FFN dimension rounding (default: 64)
        mlstm_round_up_to_multiple_of: mLSTM dimension rounding (default: 64)
        chunk_size: mLSTM chunk size (default: 64)
        kernel_mode: Kernel mode (default: "metal_chunkwise")
        norm_eps: Normalization epsilon (default: 1e-6)
        norm_reduction_force_float32: Force float32 in norm reductions (default: True)
        use_bias: Whether to use bias (default: False)
        eps: Numerical stability epsilon (default: 1e-6)
        sparsity_mask: Optional NCPS wiring sparsity mask
        
    Weight Keys (in safetensors):
        - backbone.blocks.{i}.norm_mlstm.weight
        - backbone.blocks.{i}.mlstm_layer.q.weight
        - backbone.blocks.{i}.mlstm_layer.k.weight
        - backbone.blocks.{i}.mlstm_layer.v.weight
        - backbone.blocks.{i}.mlstm_layer.igate_preact.weight
        - backbone.blocks.{i}.mlstm_layer.igate_preact.bias
        - backbone.blocks.{i}.mlstm_layer.fgate_preact.weight
        - backbone.blocks.{i}.mlstm_layer.fgate_preact.bias
        - backbone.blocks.{i}.mlstm_layer.ogate_preact.weight
        - backbone.blocks.{i}.mlstm_layer.multihead_norm.weight
        - backbone.blocks.{i}.mlstm_layer.out_proj.weight
        - backbone.blocks.{i}.norm_ffn.weight
        - backbone.blocks.{i}.ffn.proj_up.weight
        - backbone.blocks.{i}.ffn.proj_up_gate.weight
        - backbone.blocks.{i}.ffn.proj_down.weight
    """

    def __init__(
            self,
            block_index: int,
            embedding_dim: int = 4096,
            num_heads: int = 8,
            qk_dim_factor: float = 0.5,
            v_dim_factor: float = 1.0,
            gate_soft_cap: float = 15.0,
            ffn_proj_factor: float = 2.667,
            ffn_round_up_to_multiple_of: int = 64,
            mlstm_round_up_to_multiple_of: int = 64,
            chunk_size: int = 64,
            kernel_mode: str = "metal_chunkwise",
            norm_eps: float = 1e-6,
            norm_reduction_force_float32: bool = True,
            use_bias: bool = False,
            eps: float = 1e-6,
            sparsity_mask: Optional[mx.array] = None,
            compute_dtype: mx.Dtype = mx.float32,
            state_dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()

        self.block_index = block_index
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.eps = eps

        # Compute dimensions with proper rounding (matches safetensors)
        # QK dimension per head
        qk_dim_per_head_unrounded = int(embedding_dim * qk_dim_factor / num_heads)
        self.qk_dim_per_head = self._round_up_to_multiple(
            qk_dim_per_head_unrounded,
            mlstm_round_up_to_multiple_of
        )

        # V dimension per head
        v_dim_per_head_unrounded = int(embedding_dim * v_dim_factor / num_heads)
        self.v_dim_per_head = self._round_up_to_multiple(
            v_dim_per_head_unrounded,
            mlstm_round_up_to_multiple_of
        )

        # Total hidden dimension (for output projection)
        self.hidden_size = num_heads * self.v_dim_per_head

        # FFN hidden dimension with rounding
        ffn_hidden_dim_unrounded = int(embedding_dim * ffn_proj_factor)
        self.ffn_hidden_dim = self._round_up_to_multiple(
            ffn_hidden_dim_unrounded,
            ffn_round_up_to_multiple_of
        )

        # Pre-normalization for mLSTM
        self.norm_mlstm = RMSNormCell(
            dims=embedding_dim,
            eps=norm_eps,
            force_float32_reductions=norm_reduction_force_float32,
            param_dtype=compute_dtype,
        )

        # mLSTM cell
        self.mlstm_cell = mLSTMNeuron(
            input_size=embedding_dim,
            num_heads=num_heads,
            qk_dim_per_head=self.qk_dim_per_head,
            v_dim_per_head=self.v_dim_per_head,
            chunk_size=chunk_size,
            use_bias=use_bias,
            eps=eps,
            gate_soft_cap=gate_soft_cap,
            compute_dtype=compute_dtype,
            state_dtype=state_dtype,
        )

        # Pre-normalization for FFN
        self.norm_ffn = RMSNormCell(
            dims=embedding_dim,
            eps=norm_eps,
            force_float32_reductions=norm_reduction_force_float32,
            param_dtype=compute_dtype,
        )

        # FFN (SwiGLU pattern: proj_up_gate is the gate path)
        self.ffn_proj_up = nn.Linear(embedding_dim, self.ffn_hidden_dim, bias=use_bias)
        self.ffn_proj_up_gate = nn.Linear(embedding_dim, self.ffn_hidden_dim, bias=use_bias)
        self.ffn_proj_down = nn.Linear(self.ffn_hidden_dim, embedding_dim, bias=use_bias)

        # Store config knobs for inspection/loading
        self.gate_soft_cap = gate_soft_cap
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        """Round up value to nearest multiple."""
        return ((value + multiple - 1) // multiple) * multiple

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:
        """
        Forward pass through xLSTM-7B block.
        
        Architecture:
            residual = x
            x = norm_mlstm(x)
            x = mlstm_cell(x, state)  # Returns (output, new_state)
            x = x + residual  # Residual connection
            
            residual = x
            x = norm_ffn(x)
            x = ffn(x)
            x = x + residual  # Residual connection
        
        Args:
            x: Input [B, S, embedding_dim]
            state: Optional mLSTM state (C, n, m)
            
        Returns:
            (output, new_state):
                output: [B, S, embedding_dim]
                new_state: (C, n, m) from mLSTM
        """
        # === mLSTM Branch ===
        # Residual connection
        residual = x

        # Pre-norm
        x_normed = self.norm_mlstm(x)

        # mLSTM cell (handles sequence processing internally)
        mlstm_out, new_state = self.mlstm_cell(x_normed, state)

        # Residual
        x = residual + mlstm_out

        # === FFN Branch ===
        # Residual connection
        residual = x

        # Pre-norm
        x_normed = self.norm_ffn(x)

        # SwiGLU FFN: swish(proj_up_gate(x)) * proj_up(x)
        gate = nn.silu(self.ffn_proj_up_gate(x_normed))
        up = self.ffn_proj_up(x_normed)
        hidden = gate * up

        # Down projection
        ffn_out = self.ffn_proj_down(hidden)

        # Residual
        x = residual + ffn_out

        return x, new_state

    def get_weight_keys(self) -> Dict[str, str]:
        """
        Get mapping of module parameters to safetensors keys.
        
        Returns:
            Dict mapping self.parameter_path -> safetensors_key
            
        Example:
            {
                "norm_mlstm.weight": "backbone.blocks.0.norm_mlstm.weight",
                "mlstm_cell.q_proj.weight": "backbone.blocks.0.mlstm_layer.q.weight",
                ...
            }
        """
        i = self.block_index
        prefix = f"backbone.blocks.{i}"

        return {
            # mLSTM normalization
            "norm_mlstm.weight": f"{prefix}.norm_mlstm.weight",

            # mLSTM cell projections
            "mlstm_cell.projection_cell.q_proj.weight": f"{prefix}.mlstm_layer.q.weight",
            "mlstm_cell.projection_cell.k_proj.weight": f"{prefix}.mlstm_layer.k.weight",
            "mlstm_cell.projection_cell.v_proj.weight": f"{prefix}.mlstm_layer.v.weight",

            # mLSTM gates
            "mlstm_cell.projection_cell.igate_proj.weight": f"{prefix}.mlstm_layer.igate_preact.weight",
            "mlstm_cell.projection_cell.igate_proj.bias": f"{prefix}.mlstm_layer.igate_preact.bias",
            "mlstm_cell.projection_cell.fgate_proj.weight": f"{prefix}.mlstm_layer.fgate_preact.weight",
            "mlstm_cell.projection_cell.fgate_proj.bias": f"{prefix}.mlstm_layer.fgate_preact.bias",

            # Output gate projection (uses canonical ogate weight)
            "mlstm_cell.output_cell.ogate_proj.weight": f"{prefix}.mlstm_layer.ogate_preact.weight",

            # mLSTM multihead norm
            "mlstm_cell.output_cell.norm.weight": f"{prefix}.mlstm_layer.multihead_norm.weight",

            # mLSTM output projection
            "mlstm_cell.output_cell.out_proj.weight": f"{prefix}.mlstm_layer.out_proj.weight",

            # FFN normalization
            "norm_ffn.weight": f"{prefix}.norm_ffn.weight",

            # FFN projections
            "ffn_proj_up.weight": f"{prefix}.ffn.proj_up.weight",
            "ffn_proj_up_gate.weight": f"{prefix}.ffn.proj_up_gate.weight",
            "ffn_proj_down.weight": f"{prefix}.ffn.proj_down.weight",
        }

    @classmethod
    def from_config(
            cls,
            block_index: int,
            config: Dict[str, Any],
            sparsity_mask: Optional[mx.array] = None,
            **overrides,
    ) -> "mLSTMBlock":
        """
        Create cell from config.json dict.
        
        Args:
            block_index: Block index (0-31 for xLSTM-7B)
            config: Config dict from config.json
            sparsity_mask: Optional NCPS wiring sparsity mask
            
        Returns:
            Initialized mLSTMBlock
            
        Example:
            >>> import json
            >>> with open("xlstm_7b_model/config.json") as f:
            ...     config = json.load(f)
            >>> cell = mLSTMBlock.from_config(0, config)
        """
        norm_reduction_force_float32 = config.get('norm_reduction_force_float32', True)
        compute_dtype = overrides.get(
            'compute_dtype', resolve_dtype(config.get('autocast_kernel_dtype', 'float32'))
        )
        state_dtype = overrides.get(
            'state_dtype', resolve_dtype(config.get('inference_state_dtype', 'float32'))
        )

        return cls(
            block_index=block_index,
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            qk_dim_factor=config['qk_dim_factor'],
            v_dim_factor=config['v_dim_factor'],
            gate_soft_cap=config['gate_soft_cap'],
            ffn_proj_factor=config.get('ffn_proj_factor', 2.667),
            ffn_round_up_to_multiple_of=config.get('ffn_round_up_to_multiple_of', 64),
            mlstm_round_up_to_multiple_of=config.get('mlstm_round_up_to_multiple_of', 64),
            chunk_size=config.get('chunk_size', 64),
            kernel_mode=config.get('kernel_mode', 'metal_chunkwise'),
            norm_eps=config.get('norm_eps', 1e-6),
            norm_reduction_force_float32=norm_reduction_force_float32,
            use_bias=config.get('use_bias', False),
            eps=config.get('eps', 1e-6),
            sparsity_mask=sparsity_mask,
            compute_dtype=compute_dtype,
            state_dtype=state_dtype,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "block_index": self.block_index,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "qk_dim_per_head": self.qk_dim_per_head,
            "v_dim_per_head": self.v_dim_per_head,
            "hidden_size": self.hidden_size,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "gate_soft_cap": self.gate_soft_cap,
            "eps": self.eps,
        }
