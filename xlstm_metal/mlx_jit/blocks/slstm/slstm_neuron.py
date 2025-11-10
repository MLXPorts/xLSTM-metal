"""sLSTM neuron for NCPS - scalar LSTM unit.

This is a scalar LSTM neuron that can be wired together using NCPS patterns.
Uses exponential gating and scalar cell states (vs matrix states in mLSTM).
"""

from __future__ import annotations
from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .kernel import slstm_sequential


class sLSTMCell(nn.Module):
    """
    Scalar LSTM (sLSTM) cell with multi-head architecture.

    Key differences from mLSTM:
    - Scalar cell state c (vector per head, not matrix)
    - Exponential gating: i = exp(ĩ), f = exp(f̃)
    - Stabilizer m for numerical stability
    - Simpler than mLSTM but still powerful

    Args:
        input_size: Input dimension
        num_heads: Number of sLSTM heads
        head_dim: Hidden dimension per head
        use_bias: Whether to use bias in projections
        eps: Numerical stability epsilon
        gate_soft_cap: Soft cap value for gates
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        head_dim: int,
        use_bias: bool = False,
        eps: float = 1e-6,
        gate_soft_cap: float = 15.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.gate_soft_cap = gate_soft_cap

        hidden_size = num_heads * head_dim

        # Projections for z (cell input), i, f, o gates
        self.z_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.igate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.ogate_proj = nn.Linear(input_size, hidden_size, bias=use_bias)

        # Optional: recurrent connections (block-diagonal R matrices)
        # For now keeping it simple without recurrence

        # Group norm (per-head layer norm)
        # This normalizes across the head dimension
        from xlstm_metal.mlx_jit.blocks.mlstm.multihead_norm.multihead_norm import MultiHeadLayerNorm
        self.group_norm = MultiHeadLayerNorm(
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            use_weight=True,
            use_bias=False,
            force_float32_reductions=True
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, input_size, bias=use_bias)

    def soft_cap_gates(self, x: mx.array) -> mx.array:
        """Apply soft capping to gate pre-activations."""
        if self.gate_soft_cap is None:
            return x
        cap = mx.array(self.gate_soft_cap, dtype=x.dtype)
        return mx.multiply(cap, mx.tanh(mx.divide(x, cap)))

    def __call__(self, x: mx.array, state=None) -> Tuple[mx.array, Tuple]:
        """
        Process sequence through sLSTM.

        Args:
            x: Input [B, S, D]
            state: Optional tuple (c, n, m) where:
                - c: cell state [B, NH, H]
                - n: normalizer state [B, NH, H]
                - m: stabilizer [B, NH]

        Returns:
            output: Output [B, S, D]
            new_state: Tuple (c_final, n_final, m_final)
        """
        B, S, _ = x.shape

        # Project inputs
        z = self.z_proj(x)  # [B, S, NH*H]
        i_preact = self.igate_proj(x)  # [B, S, NH]
        f_preact = self.fgate_proj(x)  # [B, S, NH]
        o_preact = self.ogate_proj(x)  # [B, S, NH*H]

        # Apply soft capping to gates
        i_preact = self.soft_cap_gates(i_preact)
        f_preact = self.soft_cap_gates(f_preact)
        o_preact = self.soft_cap_gates(o_preact)

        # Reshape for multi-head processing
        # z: [B, S, NH*H] -> [B, NH, S, H]
        z = z.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # i, f: [B, S, NH] -> [B, NH, S]
        i_preact = i_preact.transpose(0, 2, 1)
        f_preact = f_preact.transpose(0, 2, 1)

        # o: [B, S, NH*H] -> [B, NH, S]  (just num_heads dimension, not per-head values)
        # Actually for sLSTM, o gate should be scalar per head
        o_gate_per_head = self.ogate_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        # Average across head_dim to get scalar per head
        o_preact_scalar = mx.mean(o_gate_per_head, axis=-1)  # [B, S, NH]
        o_preact_scalar = o_preact_scalar.transpose(0, 2, 1)  # [B, NH, S]

        # Extract initial states
        c_initial = state[0] if state else None
        n_initial = state[1] if state else None
        m_initial = state[2] if state else None

        # Process through sLSTM kernel
        # Returns h: [B, NH, S, H], states: (c, n, m)
        h, (c_final, n_final, m_final) = slstm_sequential(
            z=z,
            i_preact=i_preact,
            f_preact=f_preact,
            o_preact=o_preact_scalar,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            eps=self.eps,
            return_last_states=True
        )

        # Transpose back: [B, NH, S, H] -> [B, S, NH, H]
        h = h.transpose(0, 2, 1, 3)

        # Apply group norm (returns flattened [B, S, NH*H])
        h_norm = self.group_norm(h)

        # Output projection
        output = self.out_proj(h_norm)

        # Return output and final states
        new_state = (c_final, n_final, m_final)
        return output, new_state


__all__ = ['sLSTMCell']
