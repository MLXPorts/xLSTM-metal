"""sLSTM Stepwise Kernel Cell - pure recurrence with Metal acceleration.

This is the "during" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The stepwise kernel cell implements sequential recurrence using
Metal-accelerated kernels with canonical sLSTM equations for numerical stability.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .slstm_metal_kernel import sLSTMMetalKernel


class sLSTMStepwiseKernelCell(nn.Module):
    """
    sLSTM Stepwise Kernel Cell - sequential recurrence only.

    Implements canonical sLSTM recurrence from xlstm package with:
    - Proper stability clamps: min(exp(...), 1.0)
    - logsigmoid for forget gate
    - tanh(z) for cell input
    - Numerical stability via Metal kernels

    No projections, no output processing - pure recurrence.

    Args:
        num_heads: Number of sLSTM heads
        head_dim: Hidden dimension per head
        eps: Numerical stability epsilon
    """

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps

        # Metal kernel for recurrence
        self.metal_kernel = sLSTMMetalKernel(num_heads, head_dim, eps)

    def __call__(
            self,
            z: mx.array,
            i_preact: mx.array,
            f_preact: mx.array,
            o_preact: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:
        """
        Apply single-step sLSTM recurrence using Metal kernel.

        Args:
            z: Cell input candidate [B, NH, H]
            i_preact: Input gate pre-activation [B, NH] (soft-capped)
            f_preact: Forget gate pre-activation [B, NH] (soft-capped)
            o_preact: Output gate pre-activation [B, NH] (soft-capped)
            state: Optional previous state (c, n, m)
                   c: [B, NH, H] - cell state
                   n: [B, NH, H] - normalizer
                   m: [B, NH] - stabilizer

        Returns:
            h: Hidden states [B, NH, H]
            new_state: Updated state (c, n, m)
        """
        from .slstm_metal_kernel import slstm_step_metal

        B, NH, H = z.shape

        # Initialize state if needed
        if state is None:
            c_state = mx.zeros((B, NH, H), dtype=z.dtype)
            n_state = mx.zeros((B, NH, H), dtype=z.dtype)
            m_state = mx.zeros((B, NH), dtype=z.dtype)
        else:
            c_state, n_state, m_state = state

        # Single timestep recurrence
        h, c_new, n_new, m_new = slstm_step_metal(
            z, i_preact, f_preact, o_preact,
            c_state, n_state, m_state,
            self.eps
        )

        new_state = (c_new, n_new, m_new)
        return h, new_state


__all__ = ['sLSTMStepwiseKernelCell']
