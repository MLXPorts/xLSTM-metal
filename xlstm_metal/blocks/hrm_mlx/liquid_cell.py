#!/usr/bin/env python
"""
Liquid Time Constant (LTC) Cell for MLX.

Stable continuous-time recurrent cell with adaptive time constants.
Ported from src/lnn_hrm/liquid_time_constant.py (PyTorch) to MLX.
"""

import mlx.core as mx
import mlx.nn as nn


class LiquidTimeConstantMLX(nn.Module):
    """Stable liquid cell (LTC-style) for block-scale updates.

    Implements gated blending of short/long-term transformations with
    time-conditioned dynamics and learned tau parameters.

    Batch-first API: x [B,D], h [B,D], t scalar or [B].
    Returns (new_state, output).

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        tau_init: Initial time constant (default: 1.0)

    Example:
        >>> cell = LiquidTimeConstantMLX(input_size=512, hidden_size=512)
        >>> x = mx.random.normal((4, 512))  # Batch of 4
        >>> h = mx.zeros((4, 512))  # Initial hidden state
        >>> t = mx.array([1.0])  # Time step
        >>> h_new, output = cell(x, h, t)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau_init: float = 1.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Backbone: processes [x || h] concatenated input
        fc1_input_dim = mx.add(mx.array(input_size), mx.array(hidden_size))
        self.fc1 = nn.Linear(fc1_input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Time and state networks
        self.time_net = nn.Linear(hidden_size, hidden_size)
        self.state_net_g = nn.Linear(hidden_size, hidden_size)
        self.state_net_h = nn.Linear(hidden_size, hidden_size)

        # Learned time constant τ (strictly positive via softplus)
        tau_raw_init = mx.log(mx.subtract(mx.exp(mx.array(tau_init)), mx.array(1.0)))
        self.tau_raw = mx.multiply(mx.ones((hidden_size,)), tau_raw_init)

        # State transition matrix A
        self.A = mx.multiply(mx.random.normal((hidden_size,)), mx.array(0.02))

        # LayerNorm for stability
        self.norm = nn.LayerNorm(hidden_size)

    def __call__(
        self,
        x: mx.array,
        h: mx.array,
        t: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Forward pass through liquid cell.

        Args:
            x: Input of shape (B, input_size)
            h: Hidden state of shape (B, hidden_size)
            t: Time scalar or (B,) array

        Returns:
            h_out: New hidden state (B, hidden_size)
            output: Output (same as h_out for this cell)
        """
        # Expand t to match batch if scalar
        if t.ndim == 0:
            t = mx.broadcast_to(t, (x.shape[0],))

        # Compute time constant τ (strictly positive)
        tau = mx.add(nn.softplus(self.tau_raw), mx.array(1e-4))

        # Concatenate input and hidden state
        comb = mx.concatenate([x, h], axis=-1)

        # Backbone network with tanh nonlinearity
        feats = self.fc1(comb)
        feats = mx.tanh(feats)
        feats = self.fc2(feats)

        # Time-conditioned gating
        f_t = mx.sigmoid(self.time_net(feats))

        # Short-term and long-term state transforms
        g_x = self.state_net_g(feats)
        h_x = self.state_net_h(feats)

        # Time-modulated gate: higher t -> lower gate value
        t_expanded = t.reshape(-1, 1)  # (B, 1)
        gate = mx.sigmoid(mx.multiply(mx.negative(f_t), t_expanded))

        # Blend short and long term
        one_minus_gate = mx.subtract(mx.array(1.0), gate)
        h_new = mx.add(mx.multiply(gate, g_x), mx.multiply(one_minus_gate, h_x))

        # Residual update with stability clamping
        delta = mx.clip(mx.subtract(h_new, h), mx.array(-1.0), mx.array(1.0))
        h_out = self.norm(mx.add(h, delta))

        return h_out, h_out
