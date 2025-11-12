"""mLSTM Output Cell - handles output processing.

This is the "after" cell in the mLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The output cell transforms kernel outputs back to input space:
- Per-head normalization
- Output gate modulation
- Final linear projection

It contains NO recurrence logic.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.rms_norm import MultiHeadRMSNormCell


class mLSTMOutputCell(nn.Module):
    """
    mLSTM Output Cell - output transformation only.

    Handles all output processing for mLSTM:
    - Per-head RMS normalization
    - Output gate computation and modulation
    - Final projection back to input space

    No recurrence - just output transformations.

    Args:
        input_size: Input dimension (for x_orig and final output)
        num_heads: Number of attention heads
        v_dim_per_head: Value dimension per head
        use_bias: Whether to use bias in output gate/projection
        eps: Epsilon for RMS normalization
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            v_dim_per_head: int,
            use_bias: bool = False,
            eps: float = 1e-6,
            force_float32_reductions: bool = True,
            param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.v_dim_per_head = v_dim_per_head
        v_dim = num_heads * v_dim_per_head

        # Multi-head RMS normalization (per-head)
        self.norm = MultiHeadRMSNormCell(
            num_heads=num_heads,
            head_dim=v_dim_per_head,
            eps=eps,
            force_float32_reductions=force_float32_reductions,
            param_dtype=param_dtype,
        )

        # Output gate projection (from original input)
        self.ogate_proj = nn.Linear(input_size, v_dim, bias=use_bias)

        # Final output projection
        self.out_proj = nn.Linear(v_dim, input_size, bias=use_bias)

    def __call__(
            self,
            h: mx.array,
            x_orig: mx.array
    ) -> mx.array:
        """
        Process kernel output to final output.

        Args:
            h: Hidden states from kernel [B, NH, S, DH_v]
            x_orig: Original input (for output gate) [B, S, input_size]

        Returns:
            output: Final output [B, S, input_size]
        """
        B, NH, S, DH_v = h.shape

        # Transpose back: [B, NH, S, DH_v] -> [B, S, NH, DH_v]
        h = h.transpose(0, 2, 1, 3)

        # Normalize (returns flattened [B, S, NH*DH_v])
        h_norm = self.norm(h)

        # Output gate (from original input, flattened to match h_norm)
        o_gate = mx.sigmoid(self.ogate_proj(x_orig))  # [B, S, NH*DH_v]

        # Gate the normalized hidden states
        h_gated = mx.multiply(h_norm, o_gate)

        # Final output projection
        output = self.out_proj(h_gated)  # [B, S, input_size]

        return output


__all__ = ['mLSTMOutputCell']
