"""mLSTM Neuron - wires together projection, kernel, and output cells.

The neuron is the complete mLSTM layer that wires together:
    Input → Projection Cell → Kernel Cell → Output Cell → Output

The neuron owns the dispatch logic (parallel vs recurrent mode)
and composes the before/during/after cells.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .mlstm_projection_cell import mLSTMProjectionCell
from .mlstm_parallel_kernel_cell import mLSTMParallelKernelCell
from .mlstm_recurrent_kernel_cell import mLSTMRecurrentKernelCell
from .mlstm_output_cell import mLSTMOutputCell


class mLSTMNeuron(nn.Module):
    """
    mLSTM Neuron - complete mLSTM layer with dispatch logic.

    Wires together the mLSTM pipeline:
    1. Projection Cell: x → q, k, v, gates
    2. Kernel Cell: q, k, v, gates, state → h, new_state (dispatched)
    3. Output Cell: h, x → output

    The neuron handles kernel dispatch (parallel vs recurrent)
    and composes the modular cells.

    Args:
        input_size: Input dimension (embedding_dim)
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head
        v_dim_per_head: Value dimension per head
        chunk_size: Chunk size for parallel kernel
        kernel_mode: Kernel mode ('parallel' or 'recurrent')
        use_bias: Whether to use bias in projections
        eps: Numerical stability epsilon
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            qk_dim_per_head: int,
            v_dim_per_head: int,
            chunk_size: int = 64,
            kernel_mode: str = "parallel",
            use_bias: bool = False,
            eps: float = 1e-6,
            gate_soft_cap: Optional[float] = None,
            compute_dtype: mx.Dtype = mx.float32,
            state_dtype: mx.Dtype = mx.float32,
            force_float32_reductions: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.chunk_size = chunk_size
        self.kernel_mode = kernel_mode
        self.eps = eps
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype
        self.force_float32_reductions = force_float32_reductions

        # Validate kernel mode
        valid_modes = {"parallel", "recurrent"}
        if kernel_mode not in valid_modes:
            raise ValueError(
                f"Invalid kernel_mode '{kernel_mode}'. "
                f"Valid options: {sorted(valid_modes)}"
            )

        # === Before Cell: Projections ===
        self.projection_cell = mLSTMProjectionCell(
            input_size=input_size,
            num_heads=num_heads,
            qk_dim_per_head=qk_dim_per_head,
            v_dim_per_head=v_dim_per_head,
            use_bias=use_bias,
            gate_soft_cap=gate_soft_cap,
        )

        # === During Cells: Kernel dispatch ===
        self.parallel_kernel = mLSTMParallelKernelCell(
            num_heads=num_heads,
            qk_dim_per_head=qk_dim_per_head,
            v_dim_per_head=v_dim_per_head,
            chunk_size=chunk_size,
            eps=eps,
            compute_dtype=compute_dtype,
            state_dtype=state_dtype,
        )

        self.recurrent_kernel = mLSTMRecurrentKernelCell(
            num_heads=num_heads,
            qk_dim_per_head=qk_dim_per_head,
            v_dim_per_head=v_dim_per_head,
            eps=eps,
            compute_dtype=compute_dtype,
            state_dtype=state_dtype,
        )

        # === After Cell: Output processing ===
        self.output_cell = mLSTMOutputCell(
            input_size=input_size,
            num_heads=num_heads,
            v_dim_per_head=v_dim_per_head,
            use_bias=use_bias,
            eps=eps,
            force_float32_reductions=force_float32_reductions,
            param_dtype=compute_dtype,
        )

    @property
    def state_size(self) -> Tuple[int, int, int]:
        """Return state dimensions (C, n, m)."""
        return (
            self.num_heads * self.qk_dim_per_head * self.v_dim_per_head,
            self.num_heads * self.qk_dim_per_head,
            self.num_heads
        )

    @property
    def output_size(self) -> int:
        """Return output dimension."""
        return self.input_size

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:
        """
        Forward pass through complete mLSTM neuron.

        Args:
            x: Input [B, S, input_size]
            state: Optional previous state (C, n, m)

        Returns:
            output: Output [B, S, input_size]
            new_state: Updated state (C, n, m)
        """
        # === Before: Project to Q/K/V and gates ===
        q, k, v, i_preact, f_preact = self.projection_cell(x)
        q = mx.array(q, dtype=self.compute_dtype)
        k = mx.array(k, dtype=self.compute_dtype)
        v = mx.array(v, dtype=self.compute_dtype)
        i_preact = mx.array(i_preact, dtype=self.compute_dtype)
        f_preact = mx.array(f_preact, dtype=self.compute_dtype)

        # === During: Apply kernel (dispatch based on mode) ===
        if self.kernel_mode == "parallel":
            h, new_state = self.parallel_kernel(
                q, k, v, i_preact, f_preact, state
            )
        elif self.kernel_mode == "recurrent":
            h, new_state = self.recurrent_kernel(
                q, k, v, i_preact, f_preact, state
            )
        else:
            raise ValueError(f"Unknown kernel_mode: {self.kernel_mode}")

        # === After: Process output ===
        output = self.output_cell(h, x)

        return output, new_state

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "num_heads": self.num_heads,
            "qk_dim_per_head": self.qk_dim_per_head,
            "v_dim_per_head": self.v_dim_per_head,
            "chunk_size": self.chunk_size,
            "kernel_mode": self.kernel_mode,
            "eps": self.eps,
        }


__all__ = ['mLSTMNeuron']
