"""mLSTM Projection Cell – MLX Implementation (Before Phase)

Overview
--------
The projection cell is the **"before"** component in the modular mLSTM
pipeline. It receives raw input embeddings and produces:
  1. Query (Q), Key (K), Value (V) tensors reshaped for multi-head processing
  2. Input gate (i) and forget gate (f) preactivations (pre-sigmoid/softplus)

It contains **no recurrence** and **no output gating**—purely feedforward
transformations preparing inputs for the kernel cell.

Pipeline Position
-----------------
Input [B, S, D]
  → Projection Cell → (q, k, v, i_preact, f_preact)
  → Kernel Cell     → hidden states h
  → Output Cell     → final output [B, S, D]

Tensor Shapes
-------------
Inputs:
  x : [B, S, input_size]

Outputs:
  q        : [B, NH, S, DH_qk]   query (multi-head reshaped)
  k        : [B, NH, S, DH_qk]   key   (multi-head reshaped)
  v        : [B, NH, S, DH_v]    value (multi-head reshaped)
  i_preact : [B, NH, S]          input gate preactivation
  f_preact : [B, NH, S]          forget gate preactivation

Gate Preactivations
-------------------
Gates are projected to [B, S, NH] then transposed to [B, NH, S] for
per-head processing. Optional soft-cap (cap * tanh(x / cap)) bounds
magnitudes before exponential operations in the kernel cell.

Why Separate i/f Gates?
------------------------
Input and forget gates control the contribution of new vs old memory:
  - i_preact → sigmoid → weight for new content (k ⊗ v)
  - f_preact → sigmoid → weight for prior state C_{t-1}
Exponential stabilization (log-space) happens in the kernel cell.

Use Bias?
---------
Q/K/V projections typically do **not** use bias (canonical xLSTM setting).
Gate projections (igate_proj, fgate_proj) **do** use bias (initialized to
reasonable defaults) to allow learned baseline gating behavior.

Soft-Cap
--------
If `gate_soft_cap` is set, applies cap * tanh(preact / cap) to i_preact
and f_preact. This prevents extreme values that could destabilize the
exponential gating in the kernel phase.

NCPS Terminology
----------------
In NCPS / liquid time-constant networks:
  - Q/K/V projections are "feature groups" (different representational subspaces)
  - i/f gates are "excitatory/inhibitory" control signals
This projection cell follows that modular pattern.

Parity
------
Logic mirrors torch-native `mLSTMProjectionCell` for cross-backend testing.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.blocks.soft_cap.softcap import SoftCapCell


class mLSTMProjectionCell(nn.Module):
    """Input projection stage for mLSTM (no recurrence, no output processing).

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
    use_bias : bool, default False
        Whether Q/K/V linear layers include bias (typically False).
    gate_soft_cap : float | None, optional
        Soft-cap value for gate preactivations (None disables).
    soft_cap_cell : SoftCapCell | None, optional
        Custom soft-cap cell instance (default creates new SoftCapCell).

    Returns (forward)
    -----------------
    q : mx.array [B, NH, S, DH_qk]
    k : mx.array [B, NH, S, DH_qk]
    v : mx.array [B, NH, S, DH_v]
    i_preact : mx.array [B, NH, S]
    f_preact : mx.array [B, NH, S]
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            qk_dim_per_head: int,
            v_dim_per_head: int,
            use_bias: bool = False,
            gate_soft_cap: Optional[float] = None,
            soft_cap_cell: Optional[SoftCapCell] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.use_bias = use_bias
        self.gate_soft_cap = gate_soft_cap
        self.soft_cap = soft_cap_cell or SoftCapCell()

        qk_dim = num_heads * qk_dim_per_head
        v_dim = num_heads * v_dim_per_head

        # Q/K/V projections (feature groups in NCPS terminology)
        self.q_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.v_proj = nn.Linear(input_size, v_dim, bias=use_bias)

        # Input/forget gates (excitatory/inhibitory neurons)
        # Gates always have bias=True for proper initialization
        self.igate_proj = nn.Linear(input_size, num_heads)
        self.fgate_proj = nn.Linear(input_size, num_heads)

    def __call__(
            self,
            x: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:  # noqa: D401
        """Project input to multi-head Q/K/V and gate preactivations.

        Parameters
        ----------
        x : mx.array [B, S, input_size]
            Input embedding sequence.

        Returns
        -------
        q : mx.array [B, NH, S, DH_qk]
            Query tensor for attention-like memory indexing.
        k : mx.array [B, NH, S, DH_qk]
            Key tensor for memory content addressing.
        v : mx.array [B, NH, S, DH_v]
            Value tensor for memory payload.
        i_preact : mx.array [B, NH, S]
            Input gate preactivation (before sigmoid).
        f_preact : mx.array [B, NH, S]
            Forget gate preactivation (before sigmoid).
        """
        B, S, _ = x.shape

        # Project to Q/K/V
        q = self.q_proj(x)  # [B, S, NH*DH_qk]
        k = self.k_proj(x)  # [B, S, NH*DH_qk]
        v = self.v_proj(x)  # [B, S, NH*DH_v]

        # Reshape to multi-head: [B, S, NH, DH] -> [B, NH, S, DH]
        q = q.reshape(B, S, self.num_heads, self.qk_dim_per_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.qk_dim_per_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.v_dim_per_head).transpose(0, 2, 1, 3)

        # Gate pre-activations: [B, S, NH] -> [B, NH, S]
        i_preact = self.igate_proj(x).transpose(0, 2, 1)
        f_preact = self.fgate_proj(x).transpose(0, 2, 1)

        if self.gate_soft_cap is not None:
            cap_tensor = mx.array(self.gate_soft_cap, dtype=i_preact.dtype)
            i_preact = mx.array(self.soft_cap(i_preact, cap_tensor), dtype=i_preact.dtype)
            f_preact = mx.array(self.soft_cap(f_preact, cap_tensor), dtype=f_preact.dtype)

        return q, k, v, i_preact, f_preact


__all__ = ['mLSTMProjectionCell']
