"""mLSTM Block (xLSTM-7B) – MLX Implementation

Overview
--------
The mLSTM block is the *matrix memory* component of xLSTM. Each attention
head maintains matrix-valued state C plus accompanying normalizers (n, m)
allowing long-range sequence modeling with stabilized exponential updates.

Composition
-----------
This high-level block wraps three conceptual cells:
  1. Projection Cell (before): input -> Q,K,V + gate preactivations
  2. Kernel Cell     (during): chunkwise or recurrent memory updates
  3. Output Cell     (after) : per-head normalization + gating + projection

The block then appends a SwiGLU feed-forward network (FFN) with its own
pre-normalization (RMSNorm). Two residual connections are applied:
  (a) input + mLSTM output
  (b) intermediate + FFN output

Sequence of Operations
----------------------
residual = x
x_normed = norm_mlstm(x)
mlstm_out, state = mlstm_cell(x_normed, state)
x = residual + mlstm_out

residual = x
x_normed = norm_ffn(x)
ffn_hidden = silu(proj_up_gate(x_normed)) * proj_up(x_normed)
ffn_out    = proj_down(ffn_hidden)
x = residual + ffn_out

State Structure (returned by mlstm_cell)
---------------------------------------
state = (C, n, m)
  C : [B, NH, DH_qk, DH_v]  memory matrices per head
  n : [B, NH, DH_qk]        normalizer vector per head
  m : [B, NH]               stabilized scalar log-sum per head

Dimension Rounding
------------------
To match pretrained safetensors, QK and V per-head dims are rounded up to
`mlstm_round_up_to_multiple_of`, and FFN hidden dim to `ffn_round_up_to_multiple_of`.

SwiGLU FFN
----------
Applies gate = silu(W_gate x) and up = W_up x then elementwise gate * up
followed by a down projection. This is standard for modern transformer blocks.

Numeric Stability
-----------------
- Norm reductions may force float32 (`norm_reduction_force_float32`) to
  reduce precision loss for mixed dtype kernels.
- `eps` ensures denominator safety in normalization and gating steps.

Metal Kernels
-------------
`kernel_mode` selects specialized MLX Metal kernels for parallel chunkwise
execution (`metal_chunkwise`) or fallback recurrent paths.

Parity & Torch Backend
----------------------
Logic mirrors the torch_native `mLSTMBlock` to enable forward parity tests
across backends.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_chunkwise.mlstm_neuron import mLSTMNeuron
from xlstm_metal.mlx_jit.blocks.rms_norm import RMSNormCell
from xlstm_metal.mlx_jit.utils import resolve_dtype


class mLSTMBlock(nn.Module):
    """High-level mLSTM + FFN block (matrix memory + SwiGLU) with residuals.

    Wraps projection, kernel, and output processing for mLSTM plus the
    feed-forward network. Provides weight key mapping for safetensors
    loading and configuration introspection.

    Parameters
    ----------
    block_index : int
        Index of the block inside the backbone (0-based).
    embedding_dim : int, default 4096
        Model embedding dimension.
    num_heads : int, default 8
        Number of attention heads.
    qk_dim_factor : float, default 0.5
        Proportion of embedding_dim allocated to Q/K per head (pre-rounding).
    v_dim_factor : float, default 1.0
        Proportion for V per head (commonly 1.0 in xLSTM).
    gate_soft_cap : float, default 15.0
        Soft cap applied to gate preactivations inside cells.
    ffn_proj_factor : float, default 2.667
        Expansion ratio for FFN hidden dimension before rounding.
    ffn_round_up_to_multiple_of : int, default 64
        Alignment multiple for FFN hidden dimension.
    mlstm_round_up_to_multiple_of : int, default 64
        Alignment multiple for QK / V per-head dims.
    chunk_size : int, default 64
        Chunk size for parallel kernel execution.
    kernel_mode : str, default "metal_chunkwise"
        Execution strategy/preset for kernels.
    norm_eps : float, default 1e-6
        Epsilon for RMSNorm stability.
    norm_reduction_force_float32 : bool, default True
        Force float32 accumulation in norm reductions.
    use_bias : bool, default False
        Whether linear layers use bias (weights are typically no-bias).
    eps : float, default 1e-6
        Numeric epsilon for internal gating/normalization.
    sparsity_mask : Optional[mx.array]
        Optional block-level sparsity wiring mask.
    compute_dtype : mx.Dtype, default mx.float32
        Dtype for forward activations.
    state_dtype : mx.Dtype, default mx.float32
        Dtype for recurrent state tensors (can differ for memory footprint).

    Returns (forward)
    -----------------
    output : mx.array [B, S, embedding_dim]
    new_state : (C, n, m) recurrent state tuple.

    Notes
    -----
    - Residual connections use pre-norm design.
    - Weight tying / LM head applied outside this block.
    - Compatible with automatic wiring / NCPS model assembly.
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
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.mlstm_round_up_to_multiple_of = mlstm_round_up_to_multiple_of
        self.chunk_size = chunk_size
        self.kernel_mode = kernel_mode
        self.norm_eps = norm_eps
        self.norm_reduction_force_float32 = norm_reduction_force_float32
        self.use_bias = use_bias
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype

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
            force_float32_reductions=norm_reduction_force_float32,
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
        """Forward pass through composite mLSTM block.

        Parameters
        ----------
        x : mx.array [B, S, embedding_dim]
            Input activations for this block.
        state : tuple | None
            Previous mLSTM state (C, n, m) or None to initialize internally.

        Returns
        -------
        output : mx.array [B, S, embedding_dim]
            Activation after mLSTM + FFN + residual pathways.
        new_state : (C, n, m)
            Updated recurrent state from mLSTM kernel cell.
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
        norm_reduction_force_float32 = overrides.get(
            'norm_reduction_force_float32', config.get('norm_reduction_force_float32', True)
        )
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
            "qk_dim_factor": self.qk_dim_factor,
            "v_dim_factor": self.v_dim_factor,
            "hidden_size": self.hidden_size,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "ffn_proj_factor": self.ffn_proj_factor,
            "ffn_round_up_to_multiple_of": self.ffn_round_up_to_multiple_of,
            "mlstm_round_up_to_multiple_of": self.mlstm_round_up_to_multiple_of,
            "chunk_size": self.chunk_size,
            "kernel_mode": self.kernel_mode,
            "gate_soft_cap": self.gate_soft_cap,
            "norm_eps": self.norm_eps,
            "norm_reduction_force_float32": self.norm_reduction_force_float32,
            "use_bias": self.use_bias,
            "eps": self.eps,
            "compute_dtype": self.compute_dtype,
            "state_dtype": self.state_dtype,
        }
