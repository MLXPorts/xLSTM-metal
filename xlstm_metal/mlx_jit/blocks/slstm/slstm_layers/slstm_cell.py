"""
Scalar LSTM (sLSTM) Cell – MLX Implementation (single-step)

Overview
--------
The scalar LSTM (sLSTM) layer is the *scalar-memory* component of xLSTM
(see Appendix A of the xLSTM paper: https://arxiv.org/pdf/2405.04517).
Instead of storing matrix-valued memory like mLSTM, each head maintains
lightweight scalar exponential statistics that enable long-range retention
with numerically stable gating.

NCPS Design Pattern
-------------------
Following NCPS / CfC style, ALL trainable parameters (projections, optional
causal conv pre-processing, group / per‑head norm, output projection) are
contained in this cell. The cell processes **one timestep** at a time:
    __call__(inputs, hx, ts) -> (output, new_hx)

Computation Flow (per timestep)
-------------------------------
1. (Optional) Causal 1D convolution over the *current* token (implemented
   via padding + conv + SiLU) for temporal mixing of i,f gate preactivations.
2. Linear projections produce:
      z_t  : candidate content         [B, NH * H]
      i_t  : input  gate preactivation [B, NH]
      f_t  : forget gate preactivation [B, NH]
      o_t  : output gate preactivation [B, NH]
3. Soft cap (cap * tanh(x / cap)) optionally applied to i_t, f_t, o_t to
   bound magnitude and stabilize exponentials.
4. Exponential stabilization:
      m_t = max(f_t + m_{t-1}, i_t)
   ensures denominators remain well‑scaled and avoids overflow/underflow.
5. Normalized gates:
      i_gate = exp(i_t              - m_t)
      f_gate = exp(f_t + m_{t-1}    - m_t)
6. State updates (per head, elementwise in H):
      c_t = f_gate * c_{t-1} + i_gate * z_t
      n_t = f_gate * n_{t-1} + i_gate
7. Normalized hidden content:
      h_tilde = c_t / (n_t + eps)
      h      = sigmoid(o_t) * h_tilde
8. Group / per‑head normalization (multi‑head layer norm) applied to h,
   flattened to [B, NH * H], then projected back to input_size.

Shapes
------
Inputs:
    inputs : [B, D]
State (hx): (c, n, m)
    c : [B, NH, H]  (content accumulator)
    n : [B, NH, H]  (normalizer accumulator)
    m : [B, NH]     (stabilizer log‑scale)
Output:
    output : [B, D]
    new_hx : (c_new, n_new, m_new)

Arguments
---------
input_size : int
    Feature dimension D.
num_heads : int
    Number of scalar heads (NH).
head_dim : int
    Per‑head hidden size (H) so NH * H = hidden_size.
conv1d_kernel_size : int, default 4
    Enables causal temporal conv for i,f gate preactivations when > 0.
use_bias : bool, default False
    Bias term in linear projections.
eps : float, default 1e-6
    Numerical stability constant for normalization.
gate_soft_cap : float, default 15.0
    Soft cap value; if None disables tanh capping.

Why Soft Capping?
-----------------
Large gate magnitudes can explode the stabilized exponentials. The soft
cap keeps preactivations in a smooth but bounded range without hard clipping.

Autograd / Numerical Notes
--------------------------
The stabilized form using m_t avoids computing exp of large positive
numbers while preserving correct ratios. Dividing by n_t + eps normalizes
cumulative weighted content, preventing scale drift.

Parity with Torch Version
-------------------------
Logic mirrors the torch_native sLSTMCell so forward parity tests can
assert closeness between MLX and PyTorch backends.
"""

from __future__ import annotations
from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn


class sLSTMCell(nn.Module):
    """Single‑timestep scalar LSTM (sLSTM) recurrence cell (MLX backend).

    Implements one autoregressive timestep with stabilized exponential gating
    and per‑head normalization. Encapsulates projections, (optional) conv
    preprocessing, normalization, and output projection.

    Forward Signature
    -----------------
    __call__(inputs, hx=None, ts=None) -> (output, new_hx)

    Parameters (see module docstring for detailed semantics)
    -------------------------------------------------------
    input_size : int
    num_heads : int
    head_dim : int
    conv1d_kernel_size : int, default 4
    use_bias : bool, default False
    eps : float, default 1e-6
    gate_soft_cap : float, default 15.0

    Returns
    -------
    output : mx.array [B, D]
        Projected hidden representation for the timestep.
    new_hx : (c_new, n_new, m_new)
        Updated recurrent states.
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            head_dim: int,
            conv1d_kernel_size: int = 4,
            use_bias: bool = False,
            eps: float = 1e-6,
            gate_soft_cap: float = 15.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv1d_kernel_size = conv1d_kernel_size
        self.eps = eps
        self.gate_soft_cap = gate_soft_cap

        hidden_size = num_heads * head_dim

        # Optional Conv1d for temporal context (applied to i, f gates only)
        if conv1d_kernel_size > 0:
            # Causal padding for temporal causality
            self.causal_pad = conv1d_kernel_size - 1
            self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=conv1d_kernel_size)
            self.conv_act = nn.SiLU()
        else:
            self.conv1d = None

        # Gate projections (canonical uses LinearHeadwiseExpand, we use Linear for now)
        # i, f use conv'd input; z, o use raw input
        self.z_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.igate_proj = nn.Linear(input_size, num_heads)
        self.fgate_proj = nn.Linear(input_size, num_heads)
        self.ogate_proj = nn.Linear(input_size, num_heads)

        # Group norm (per-head layer norm)
        from xlstm_metal.mlx_jit.blocks.mlstm.multihead_norm.multihead_norm import MultiHeadLayerNorm
        self.group_norm = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim, eps=eps)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, input_size, bias=use_bias)

    @property
    def state_size(self) -> int:
        """Total state size across all heads."""
        return self.num_heads * self.head_dim

    @property
    def output_size(self) -> int:
        """Output size (same as input_size after out_proj)."""
        return self.input_size

    def soft_cap_gates(self, x: mx.array) -> mx.array:  # noqa: D401 - detailed in module docstring
        """Apply soft capping (cap * tanh(x / cap)) to gate preactivations if enabled."""
        if self.gate_soft_cap is None:
            return x
        cap = mx.array(self.gate_soft_cap, dtype=x.dtype)
        return mx.multiply(cap, mx.tanh(mx.divide(x, cap)))

    def __call__(
            self,
            inputs: mx.array,
            hx: Optional[Tuple[mx.array, mx.array, mx.array]] = None,
            ts: Optional[float | mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:  # noqa: D401 - expanded below
        """Run one sLSTM timestep.

        Parameters
        ----------
        inputs : mx.array [B, D]
            Current timestep features.
        hx : tuple | None
            Previous state (c, n, m) or None for zero initialization.
        ts : float | mx.array | None
            Optional timestep placeholder (kept for NCPS API symmetry; unused).

        Returns
        -------
        output : mx.array [B, D]
            Hidden representation after gating, normalization, and projection.
        new_hx : (c_new, n_new, m_new)
            Updated recurrent states for next step.
        """
        B = inputs.shape[0]
        NH = self.num_heads
        H = self.head_dim

        # Apply Conv1d if enabled
        if self.conv1d is not None:
            # Add sequence dimension for Conv1d: [B, D] -> [B, 1, D]
            x_seq = mx.expand_dims(inputs, axis=1)

            # Causal padding
            if self.causal_pad > 0:
                padding = mx.zeros((B, self.causal_pad, self.input_size), dtype=inputs.dtype)
                x_seq = mx.concatenate([padding, x_seq], axis=1)

            # Apply conv and activation
            x_conv = self.conv_act(self.conv1d(x_seq))

            # Extract single timestep: [B, S, D] -> [B, D]
            x_conv = x_conv[:, -1, :]  # Last timestep
        else:
            x_conv = inputs

        # Project gates
        z_t = self.z_proj(inputs)  # [B, NH*H] - raw input
        i_t = self.igate_proj(x_conv)  # [B, NH] - conv'd input
        f_t = self.fgate_proj(x_conv)  # [B, NH] - conv'd input
        o_t = self.ogate_proj(inputs)  # [B, NH] - raw input

        # Apply soft capping
        i_t = self.soft_cap_gates(i_t)
        f_t = self.soft_cap_gates(f_t)
        o_t = self.soft_cap_gates(o_t)

        # Reshape for recurrence
        z = z_t.reshape(B, NH, H)  # [B, NH, H]

        # Initialize state if needed
        if hx is None:
            c_state = mx.zeros((B, NH, H), dtype=inputs.dtype)
            n_state = mx.zeros((B, NH, H), dtype=inputs.dtype)
            m_state = mx.zeros((B, NH), dtype=inputs.dtype)
        else:
            c_state, n_state, m_state = hx

        # Recurrence equations (exponential gating)
        # m_t = max(f̃_t + m_{t-1}, ĩ_t)
        m_new = mx.maximum(
            mx.add(f_t, m_state),
            i_t
        )  # [B, NH]

        # Exponential gates (stabilized)
        i_gate = mx.exp(mx.subtract(i_t, m_new))  # [B, NH]
        f_gate = mx.exp(mx.subtract(mx.add(f_t, m_state), m_new))  # [B, NH]
        o_gate = mx.sigmoid(o_t)  # [B, NH]

        # Expand for broadcasting
        i_expanded = mx.expand_dims(i_gate, axis=-1)  # [B, NH, 1]
        f_expanded = mx.expand_dims(f_gate, axis=-1)  # [B, NH, 1]
        o_expanded = mx.expand_dims(o_gate, axis=-1)  # [B, NH, 1]

        # Update states
        c_new = mx.add(
            mx.multiply(f_expanded, c_state),
            mx.multiply(i_expanded, z)
        )  # [B, NH, H]

        n_new = mx.add(
            mx.multiply(f_expanded, n_state),
            i_expanded
        )  # [B, NH, H]

        # Normalized hidden state
        h_tilde = mx.divide(
            c_new,
            mx.add(n_new, mx.array(self.eps, dtype=inputs.dtype))
        )  # [B, NH, H]

        # Apply output gate
        h = mx.multiply(o_expanded, h_tilde)  # [B, NH, H]

        # Apply group norm and output projection
        # group_norm expects [B, 1, NH, H]
        h_norm_input = mx.expand_dims(h, axis=1)
        h_norm = self.group_norm(h_norm_input)  # [B, 1, NH*H]
        h_norm = mx.squeeze(h_norm, axis=1)  # [B, NH*H]

        # Output projection
        output = self.out_proj(h_norm)  # [B, D]

        # Return output and new state
        new_hx = (c_new, n_new, m_new)
        return output, new_hx


__all__ = ['sLSTMCell']
