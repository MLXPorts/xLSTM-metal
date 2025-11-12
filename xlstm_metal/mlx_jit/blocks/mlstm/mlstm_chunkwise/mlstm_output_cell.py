"""mLSTM Output Cell – MLX Implementation (After Phase)

Overview
--------
The output cell is the **"after"** component in the modular mLSTM pipeline.
It receives hidden states h from the kernel cell and the original input x,
then produces the final block output via:
  1. Per-head RMS normalization of h (flattened across heads)
  2. Output gating conditioned on the original input x
  3. Final linear projection back to input_size

It contains **no recurrence**—purely feedforward post-processing.

Pipeline Position
-----------------
Input x [B, S, D]
  → Projection Cell → (q, k, v, i_preact, f_preact)
  → Kernel Cell     → hidden h [B, NH, S, DH_v]
  → Output Cell     → output [B, S, D]

Tensor Shapes
-------------
Inputs:
  h      : [B, NH, S, DH_v]   hidden states from kernel cell
  x_orig : [B, S, D]          original input (for output gate conditioning)

Output:
  output : [B, S, D]          final processed representation

Per-Head RMS Normalization
---------------------------
RMS norm is applied per head over the head_dim axis, then results are
flattened to [B, S, NH * DH_v]. This preserves per-head statistics while
enabling a shared weight vector across all heads.

Output Gate (o_gate)
--------------------
The output gate is computed from the **original input** x_orig (before
projection or recurrence), not from the hidden states. This allows the
model to conditionally attenuate or amplify memory-derived features based
on the current input context:
  o_gate = sigmoid(W_o @ x_orig)
  h_gated = h_norm ⊙ o_gate
This is analogous to the output gate in standard LSTM but conditioned on
the embedding rather than recurrent state.

Why Use x_orig?
---------------
Using the original input for the output gate provides a **skip connection**
pattern where the model can selectively bypass the recurrent memory updates
if the input context suggests they're not relevant. This improves gradient
flow and allows more flexible gating.

Final Projection
----------------
After gating, a linear layer projects [B, S, NH * DH_v] → [B, S, D], matching
the input dimensionality for residual addition at the block level.

Force Float32 Reductions
------------------------
The norm cell may internally cast to float32 during mean/variance computation
(controlled by `force_float32_reductions`) to avoid precision loss in
mixed-precision settings (e.g., bfloat16 inference).

NCPS Terminology
----------------
In NCPS / liquid time-constant networks:
  - Output gate is the "motor neuron gate" (final control signal)
  - Normalization stabilizes "inter-layer dynamics"
This cell follows that modular, composable pattern.

Parity
------
Logic mirrors torch-native `mLSTMOutputCell` for cross-backend testing.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.rms_norm import MultiHeadRMSNormCell


class mLSTMOutputCell(nn.Module):
    """Output post-processing stage for mLSTM (no recurrence, just transformation).

    Parameters
    ----------
    input_size : int
        Embedding / model dimension D.
    num_heads : int
        Number of attention heads (NH).
    v_dim_per_head : int
        Value dimension per head.
    use_bias : bool, default False
        Whether output gate and projection layers include bias.
    eps : float, default 1e-6
        Epsilon for RMS normalization stability.
    force_float32_reductions : bool, default True
        Force float32 in norm reductions for numerical stability.
    param_dtype : mx.Dtype, default mx.float32
        Dtype for norm parameters (weight).

    Returns (forward)
    -----------------
    output : mx.array [B, S, D]
        Final output after normalization, gating, and projection.
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
    ) -> mx.array:  # noqa: D401
        """Transform kernel hidden states to final output.

        Parameters
        ----------
        h : mx.array [B, NH, S, DH_v]
            Hidden states from the kernel cell (memory-integrated features).
        x_orig : mx.array [B, S, D]
            Original input embedding (used for output gate conditioning).

        Returns
        -------
        output : mx.array [B, S, D]
            Final output ready for residual addition at block level.
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
