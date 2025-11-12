"""mLSTM Neuron – MLX Implementation (Projection + Kernel + Output)

Overview
--------
The mLSTM neuron is the *per-block* composite that wires the three phases
of matrix-memory processing:
  1. Projection Cell (before): input → Q, K, V + gate preactivations
  2. Kernel Cell     (during): recurrence (parallel chunkwise or sequential) producing hidden states h
  3. Output Cell     (after) : per-head normalization, output gating, final projection back to input size

It owns the dispatch decision between chunkwise parallel kernels and
sequential recurrent kernels (reduced memory path) via `kernel_mode`.

Chunkwise vs Recurrent
----------------------
- parallel  : Uses optimized Metal kernels to process sequence in fixed-size chunks for higher throughput.
- recurrent : Processes one timestep at a time; suitable for autoregressive generation and reduced memory usage.

State Structure
---------------
State tuple = (C, n, m):
  C : [B, NH, DH_qk, DH_v]  matrix-memory outer-product accumulator
  n : [B, NH, DH_qk]        normalizer accumulator for C
  m : [B, NH]               stabilized scalar gating accumulator

Data Flow (forward)
-------------------
Input x [B, S, D]
  → projection_cell → q,k,v (reshaped) + i_preact,f_preact gates
  → kernel_cell     → hidden h [B, NH, S, DH_v], new_state (C,n,m)
  → output_cell     → output [B, S, D]

Numerical Stability
-------------------
The kernel cells internally apply stabilized exponentials (using log-space
techniques similar to sLSTM) and per-head scaling to keep activations within
well-conditioned ranges for long sequences.

Dtype Handling
--------------
Projection outputs are coerced to `compute_dtype` while recurrence state is
stored in `state_dtype`. This allows mixed precision inference while retaining
higher precision in recurrent accumulators.

Parity
------
Logic mirrors torch-native `mLSTMNeuron` for cross-backend parity testing.
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
    """Composite mLSTM layer orchestrating projection, kernel, and output phases.

    Parameters
    ----------
    input_size : int
        Embedding / model dimension D.
    num_heads : int
        Number of attention heads (NH).
    qk_dim_per_head : int
        Query/key dimension per head.
    v_dim_per_head : int
        Value dimension per head.
    chunk_size : int, default 64
        Sequence chunk length for parallel kernels.
    kernel_mode : {"parallel", "recurrent"}, default "parallel"
        Execution mode selecting chunkwise vectorized kernels or stepwise recurrence.
    use_bias : bool, default False
        Whether projection/output linear layers include bias.
    eps : float, default 1e-6
        Numerical stability constant for internal operations.
    gate_soft_cap : float | None, optional
        Soft-cap value for gate preactivations (None disables capping).
    compute_dtype : mx.Dtype, default mx.float32
        Dtype for activations and arithmetic inside kernels.
    state_dtype : mx.Dtype, default mx.float32
        Dtype for recurrent state tensors (C, n, m).
    force_float32_reductions : bool, default True
        Force float32 in reduction ops (norms, sums) for stability.

    Returns (forward)
    -----------------
    output : mx.array [B, S, D]
        Final dense representation after output processing.
    new_state : (C, n, m)
        Updated recurrent memory state tuple for next forward call.
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
        self.use_bias = use_bias
        self.gate_soft_cap = gate_soft_cap

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
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:  # noqa: D401 - detailed below
        """Run the full mLSTM pipeline for a sequence batch.

        Parameters
        ----------
        x : mx.array [B, S, D]
            Input embedding sequence.
        state : tuple | None
            Previous recurrent state (C, n, m) or None for initialization.

        Returns
        -------
        output : mx.array [B, S, D]
            Transformed output after memory integration and gating.
        new_state : (C, n, m)
            Updated memory state for next call.
        """
        # === Before: Project to Q/K/V and gates ===
        q, k, v, i_preact, f_preact = self.projection_cell(x)
        q = mx.array(q, dtype=self.compute_dtype)
        k = mx.array(k, dtype=self.compute_dtype)
        v = mx.array(v, dtype=self.compute_dtype)
        i_preact = mx.array(i_preact, dtype=self.compute_dtype)
        f_preact = mx.array(f_preact, dtype=self.compute_dtype)

        use_parallel = self.kernel_mode == "parallel" and self.chunk_size > 0
        B, NH, S, _ = q.shape

        if use_parallel:
            full_chunks = (S // self.chunk_size)
            chunk_tokens = full_chunks * self.chunk_size
        else:
            chunk_tokens = 0

        h_chunks = []
        current_state = state

        if use_parallel and chunk_tokens > 0:
            head = slice(0, chunk_tokens)
            h_parallel, current_state = self.parallel_kernel(
                q[:, :, head, :],
                k[:, :, head, :],
                v[:, :, head, :],
                i_preact[:, :, head],
                f_preact[:, :, head],
                current_state,
            )
            h_chunks.append(h_parallel)

        tail_tokens = S - chunk_tokens
        if tail_tokens > 0:
            tail = slice(chunk_tokens, S)
            h_seq, current_state = self.recurrent_kernel(
                q[:, :, tail, :],
                k[:, :, tail, :],
                v[:, :, tail, :],
                i_preact[:, :, tail],
                f_preact[:, :, tail],
                current_state,
            )
            h_chunks.append(h_seq)

        if not h_chunks:
            # No tokens processed (should not happen, but guard)
            h, new_state = self.recurrent_kernel(q, k, v, i_preact, f_preact, state)
        else:
            h = mx.concatenate(h_chunks, axis=2) if len(h_chunks) > 1 else h_chunks[0]
            new_state = current_state

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
            "use_bias": self.use_bias,
            "eps": self.eps,
            "gate_soft_cap": self.gate_soft_cap,
            "compute_dtype": self.compute_dtype,
            "state_dtype": self.state_dtype,
            "force_float32_reductions": self.force_float32_reductions,
        }


__all__ = ['mLSTMNeuron']
