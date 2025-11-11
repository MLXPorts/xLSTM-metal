"""xLSTM sLSTM Cell - NCPS-compatible wrapper for sLSTM block.

This cell wraps a complete sLSTM block (sLSTM + FFN) with proper parameter
handling for model loading from safetensors and config.json.

Follows NCPS patterns for clean composability and wiring.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.slstm.slstm_block import sLSTMNeuron


class xLSTMsLSTMCell(nn.Module):
    """
    Complete xLSTM sLSTM block cell (sLSTM + FFN) for NCPS.

    This represents one full xLSTM block with sLSTM:
        input -> norm_slstm -> sLSTM -> residual
              -> norm_ffn -> FFN -> residual -> output

    Parameters are loaded from:
    - config.json: Model hyperparameters
    - safetensors: Pretrained weights (backbone.blocks.{i}.*)

    Args:
        block_index: Block index for weight loading
        embedding_dim: Model dimension
        num_heads: Number of sLSTM heads
        head_dim: Dimension per head
        gate_soft_cap: Gate soft cap value
        ffn_proj_factor: FFN projection factor
        ffn_round_up_to_multiple_of: FFN dimension rounding
        norm_eps: Normalization epsilon
        use_bias: Whether to use bias
        eps: Numerical stability epsilon
    """

    def __init__(
            self,
            block_index: int,
            embedding_dim: int = 4096,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            gate_soft_cap: float = 15.0,
            ffn_proj_factor: float = 2.667,
            ffn_round_up_to_multiple_of: int = 64,
            norm_eps: float = 1e-6,
            use_bias: bool = False,
            eps: float = 1e-6,
            **kwargs
    ):
        super().__init__()

        self.block_index = block_index
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Compute head_dim
        if head_dim is None:
            head_dim = embedding_dim // num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim

        self.gate_soft_cap = gate_soft_cap
        self.eps = eps

        # Compute FFN hidden dim
        ffn_hidden_raw = int(embedding_dim * ffn_proj_factor)
        self.ffn_hidden_dim = ((ffn_hidden_raw + ffn_round_up_to_multiple_of - 1)
                               // ffn_round_up_to_multiple_of * ffn_round_up_to_multiple_of)

        # sLSTM layer normalization
        self.norm_slstm = nn.RMSNorm(dims=embedding_dim, eps=norm_eps)

        # sLSTM neuron (NCPS pattern: holds parameters and loops over sequence)
        self.slstm_cell = sLSTMNeuron(
            input_size=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            use_bias=use_bias,
            eps=eps,
            gate_soft_cap=gate_soft_cap
        )

        # FFN layer normalization
        self.norm_ffn = nn.RMSNorm(dims=embedding_dim, eps=norm_eps)

        # FFN projections (SwiGLU)
        self.ffn_proj_up = nn.Linear(embedding_dim, self.ffn_hidden_dim, bias=False)
        self.ffn_proj_up_gate = nn.Linear(embedding_dim, self.ffn_hidden_dim, bias=False)
        self.ffn_proj_down = nn.Linear(self.ffn_hidden_dim, embedding_dim, bias=False)

    def __call__(self, x: mx.array, state=None) -> Tuple[mx.array, Tuple]:
        """
        Forward pass through sLSTM block with residual connections.

        Args:
            x: Input [B, S, D]
            state: Optional sLSTM state (c, n, m)

        Returns:
            output: Output [B, S, D]
            new_state: Updated sLSTM state
        """
        # sLSTM with residual
        residual = x
        x_normed = self.norm_slstm(x)
        slstm_out, new_state = self.slstm_cell(x_normed, state)
        x = mx.add(residual, slstm_out)

        # FFN with residual
        residual = x
        x_normed = self.norm_ffn(x)

        # SwiGLU FFN
        gate = nn.silu(self.ffn_proj_up_gate(x_normed))
        up = self.ffn_proj_up(x_normed)
        hidden = mx.multiply(gate, up)
        ffn_out = self.ffn_proj_down(hidden)

        x = mx.add(residual, ffn_out)

        return x, new_state

    def get_weight_keys(self) -> Dict[str, str]:
        """
        Get mapping of module parameters to safetensors keys.

        Returns:
            Dict mapping self.parameter_path -> safetensors_key
        """
        i = self.block_index
        prefix = f"backbone.blocks.{i}"

        return {
            # sLSTM normalization
            "norm_slstm.weight": f"{prefix}.norm_slstm.weight",

            # sLSTM cell projections
            "slstm_cell.z_proj.weight": f"{prefix}.slstm_layer.z.weight",
            "slstm_cell.igate_proj.weight": f"{prefix}.slstm_layer.igate_preact.weight",
            "slstm_cell.igate_proj.bias": f"{prefix}.slstm_layer.igate_preact.bias",
            "slstm_cell.fgate_proj.weight": f"{prefix}.slstm_layer.fgate_preact.weight",
            "slstm_cell.fgate_proj.bias": f"{prefix}.slstm_layer.fgate_preact.bias",
            "slstm_cell.ogate_proj.weight": f"{prefix}.slstm_layer.ogate_preact.weight",

            # sLSTM group norm
            "slstm_cell.group_norm.weight": f"{prefix}.slstm_layer.group_norm.weight",

            # sLSTM output projection
            "slstm_cell.out_proj.weight": f"{prefix}.slstm_layer.out_proj.weight",

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
            **kwargs
    ) -> "xLSTMsLSTMCell":
        """
        Create cell from config.json dict.

        Args:
            block_index: Block index
            config: Config dict from config.json
            **kwargs: Additional arguments

        Returns:
            Initialized xLSTMsLSTMCell
        """
        return cls(
            block_index=block_index,
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', 4),
            head_dim=config.get('head_dim', None),
            gate_soft_cap=config.get('gate_soft_cap', 15.0),
            ffn_proj_factor=config.get('ffn_proj_factor', 2.667),
            ffn_round_up_to_multiple_of=config.get('ffn_round_up_to_multiple_of', 64),
            norm_eps=config.get('norm_eps', 1e-6),
            use_bias=config.get('use_bias', False),
            eps=config.get('eps', 1e-6),
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "block_index": self.block_index,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "gate_soft_cap": self.gate_soft_cap,
            "eps": self.eps,
        }


__all__ = ['xLSTMsLSTMCell']
