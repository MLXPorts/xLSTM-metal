"""mLSTM Recurrent Kernel Cell – MLX Implementation (Sequential Recurrence)

Overview
--------
The recurrent kernel cell is the **"during"** (recurrence) component of
the mLSTM pipeline using a **step-by-step sequential** strategy. It processes
each timestep one-at-a-time in a loop, maintaining and updating the
recurrent state (C, n, m) at each step.

This mode is primarily used for:
  - Autoregressive generation (one token at a time)
  - Inference with strict memory constraints
  - Debugging / validating chunkwise parallel implementations

Sequential vs Parallel
----------------------
- **Sequential (this cell)**: Processes S timesteps in a loop. O(S) memory,
  O(S · DH_qk · DH_v) compute. No parallelism across time.

- **Parallel (chunkwise cell)**: Processes chunks in parallel. O(S · L) memory,
  O(S · L + S/L · DH_qk · DH_v) compute. High throughput for training/prefill.

When to Use Sequential
-----------------------
- **Generation**: After initial prompt processing, generate one token at a
  time. Sequential mode uses constant memory per step.
- **Low memory**: When batch size * sequence length * hidden dims exceeds
  available memory.
- **Debugging**: Sequential loop is easier to trace and validate.

State Update Equations
----------------------
For each timestep t = 0..S-1:
  1. Stabilized gating:
       f_log = -log(1 + exp(-f_t))   # log(sigmoid(f_t))
       m_t = max(f_log + m_{t-1}, i_t)
       f_gate = exp(f_log + m_{t-1} - m_t)
       i_gate = exp(i_t - m_t)

  2. State updates (per head, matrix-valued):
       C_t = f_gate * C_{t-1} + i_gate * (k_t ⊗ v_t)
       n_t = f_gate * n_{t-1} + i_gate * k_t
       m_t already computed above

  3. Output computation:
       q_scaled = q_t / sqrt(DH_qk)
       h_num = sum_over_qk( C_t * q_scaled )
       h_den = max(|q_scaled · n_t|, exp(-m_t)) + eps
       h_t = h_num / h_den

State Structure
---------------
State tuple = (C, n, m):
  C : [B, NH, DH_qk, DH_v]  matrix-memory (rank-1 outer product accumulator)
  n : [B, NH, DH_qk]        normalizer vector (for stable denominator)
  m : [B, NH]               scalar log-stabilizer (prevents exp overflow)

Why Matrix Memory?
------------------
Unlike scalar LSTM (sLSTM) which stores per-feature scalars, mLSTM stores
a DH_qk × DH_v matrix C per head. This allows content-based addressing:
  h_t ∝ C_t @ q_t
The query q_t acts as a "key" to retrieve relevant information from memory,
similar to attention but with recurrent accumulation.

Numerical Stability
-------------------
- Forget/input gates use log-space (softplus trick) to avoid exp(large).
- Stabilizer m_t keeps denominators well-scaled across long sequences.
- Mixed precision: compute in `compute_dtype`, store state in `state_dtype`.

Parity
------
Logic mirrors torch-native `mLSTMRecurrentKernelCell` for cross-backend testing.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class mLSTMRecurrentKernelCell(nn.Module):
    """Sequential step-by-step recurrence kernel for mLSTM (no projections).

    Parameters
    ----------
    num_heads : int
        Number of attention heads (NH).
    qk_dim_per_head : int
        Query/key dimension per head.
    v_dim_per_head : int
        Value dimension per head.
    eps : float, default 1e-6
        Numerical stability epsilon.
    compute_dtype : mx.Dtype, default mx.float32
        Dtype for forward pass activations.
    state_dtype : mx.Dtype, default mx.float32
        Dtype for recurrent state storage (C, n, m).

    Returns (forward)
    -----------------
    h : mx.array [B, NH, S, DH_v]
        Hidden states for all timesteps (stacked from loop).
    new_state : (C, n, m)
        Final recurrent state after processing all S steps.
    """

    def __init__(
            self,
            num_heads: int,
            qk_dim_per_head: int,
            v_dim_per_head: int,
            eps: float = 1e-6,
            compute_dtype: mx.Dtype = mx.float32,
            state_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.eps = eps
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype

    def __call__(
            self,
            q: mx.array,
            k: mx.array,
            v: mx.array,
            i_preact: mx.array,
            f_preact: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:  # noqa: D401
        """Execute sequential mLSTM recurrence (loop over timesteps).

        Parameters
        ----------
        q : mx.array [B, NH, S, DH_qk]
            Query projections.
        k : mx.array [B, NH, S, DH_qk]
            Key projections.
        v : mx.array [B, NH, S, DH_v]
            Value projections.
        i_preact : mx.array [B, NH, S]
            Input gate preactivations.
        f_preact : mx.array [B, NH, S]
            Forget gate preactivations.
        state : tuple | None
            Previous recurrent state (C, n, m) or None for initialization.

        Returns
        -------
        h : mx.array [B, NH, S, DH_v]
            Hidden states computed sequentially for all S timesteps.
        new_state : (C, n, m)
            Final recurrent state after step S-1.
        """
        B, NH, S, DH_qk = q.shape
        DH_v = v.shape[-1]

        q = mx.array(q, dtype=self.compute_dtype)
        k = mx.array(k, dtype=self.compute_dtype)
        v = mx.array(v, dtype=self.compute_dtype)
        i_preact = mx.array(i_preact, dtype=self.compute_dtype)
        f_preact = mx.array(f_preact, dtype=self.compute_dtype)

        # Initialize state
        if state is None:
            C = mx.zeros((B, NH, DH_qk, DH_v), dtype=self.state_dtype)
            n = mx.zeros((B, NH, DH_qk), dtype=self.state_dtype)
            m = mx.zeros((B, NH), dtype=self.state_dtype)
        else:
            C, n, m = state
            C = mx.array(C, dtype=self.state_dtype)
            n = mx.array(n, dtype=self.state_dtype)
            m = mx.array(m, dtype=self.state_dtype)

        # Sequential processing
        h_list = []

        eps_tensor = mx.array(self.eps, dtype=self.compute_dtype)

        for t in range(S):
            # Extract timestep
            q_t = q[:, :, t, :]  # [B, NH, DH_qk]
            k_t = k[:, :, t, :]  # [B, NH, DH_qk]
            v_t = v[:, :, t, :]  # [B, NH, DH_v]
            i_t = i_preact[:, :, t]  # [B, NH]
            f_t = f_preact[:, :, t]  # [B, NH]

            # Stabilized exponential gates
            # m_t = max(f_log + m_{t-1}, i_t)
            one = mx.array(1.0, dtype=f_t.dtype)
            f_log = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_t)))))
            m_new = mx.maximum(mx.add(f_log, m), i_t)

            # Normalized gates
            f_gate = mx.exp(mx.subtract(mx.add(f_log, m), m_new))  # [B, NH]
            i_gate = mx.exp(mx.subtract(i_t, m_new))  # [B, NH]

            # Update state
            # C_t = f_t * C_{t-1} + i_t * (k_t ⊗ v_t)
            kv_outer = k_t[:, :, :, None] * v_t[:, :, None, :]  # [B, NH, DH_qk, DH_v]
            kv_outer_state = mx.array(kv_outer, dtype=self.state_dtype)
            f_gate_state = mx.array(f_gate, dtype=self.state_dtype)
            i_gate_state = mx.array(i_gate, dtype=self.state_dtype)
            C = f_gate_state[:, :, None, None] * C + i_gate_state[:, :, None, None] * kv_outer_state

            # n_t = f_t * n_{t-1} + i_t * k_t
            k_state = mx.array(k_t, dtype=self.state_dtype)
            n = f_gate_state[:, :, None] * n + i_gate_state[:, :, None] * k_state

            # Update m
            m = mx.array(m_new, dtype=self.state_dtype)

            # Compute output: h_t = (C_t @ q_t) / (|q·n| + exp(-m) + eps)
            q_scaled = q_t * mx.rsqrt(mx.array(self.qk_dim_per_head, dtype=q_t.dtype))

            # Numerator
            C_compute = mx.array(C, dtype=self.compute_dtype)
            h_num = mx.sum(C_compute * q_scaled[:, :, :, None], axis=2)  # [B, NH, DH_v]

            # Denominator
            n_compute = mx.array(n, dtype=self.compute_dtype)
            qn_dot = mx.sum(n_compute * q_scaled, axis=2, keepdims=True)  # [B, NH, 1]
            max_val = mx.exp(mx.negative(mx.array(m, dtype=self.compute_dtype)))[:, :, None]
            h_den = mx.maximum(mx.abs(qn_dot), max_val) + eps_tensor

            h_t = h_num / h_den  # [B, NH, DH_v]
            h_list.append(h_t)

        # Stack outputs
        h = mx.stack(h_list, axis=2)  # [B, NH, S, DH_v]

        # Final state
        new_state = (C, n, m)

        return h, new_state


__all__ = ['mLSTMRecurrentKernelCell']
