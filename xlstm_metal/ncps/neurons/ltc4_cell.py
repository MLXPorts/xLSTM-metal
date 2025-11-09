"""Simplified LTC cell mirroring the original LTC4.py behaviour, ported to MLX."""

from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from .liquid_utils import get_activation

SolverName = str


class LTC4Cell(nn.Module):
    """Dense LTC cell with selectable ODE solver (semi-implicit, explicit, rk4)."""

    def __init__(
        self,
        units: int,
        activation: SolverName | Callable[[mx.array], mx.array] = "tanh",
        solver: SolverName = "semi_implicit",
        ode_unfolds: int = 6,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if solver not in {"semi_implicit", "explicit", "rk4"}:
            raise ValueError("solver must be one of {'semi_implicit', 'explicit', 'rk4'}")
        self.units = units
        self.solver = solver
        self.ode_unfolds = max(1, int(ode_unfolds))

        if callable(activation):
            self.activation_fn: Callable[[mx.array], mx.array] = activation
        else:
            self.activation_fn = get_activation(activation)

        self.input_linear: Optional[nn.Linear] = None
        self.recurrent_linear: Optional[nn.Linear] = None
        self._input_dim: Optional[int] = None
        if input_dim is not None:
            self._ensure_params(input_dim)

    def _ensure_params(self, input_dim: int) -> None:
        if self._input_dim == input_dim and self.input_linear is not None:
            return

        self.input_linear = nn.Linear(input_dim, self.units, bias=True)
        self._glorot_init(self.input_linear)
        self.recurrent_linear = nn.Linear(self.units, self.units, bias=False)
        self._glorot_init(self.recurrent_linear)
        self._input_dim = input_dim

    def _net_input(self, inputs: mx.array, prev: mx.array) -> mx.array:
        assert self.input_linear is not None and self.recurrent_linear is not None
        net = self.input_linear(inputs)
        net += mx.matmul(prev, mx.transpose(self.recurrent_linear.weight))
        return net

    def _glorot_init(self, linear: nn.Linear) -> None:
        fan_in = linear.weight.shape[1]
        fan_out = linear.weight.shape[0]
        limit = math.sqrt(mx.divide(6.0, mx.add(fan_in, fan_out)))
        linear.weight = mx.random.uniform(low=-limit, high=limit, shape=linear.weight.shape)
        if "bias" in linear:
            linear.bias = mx.zeros(linear.bias.shape, dtype=mx.float32)

    def _semi_implicit(self, prev: mx.array, net: mx.array) -> mx.array:
        act = self.activation_fn(net)
        return mx.add(prev, mx.multiply(self.ode_unfolds, mx.subtract(act, prev)))

    def _explicit(self, prev: mx.array, net: mx.array) -> mx.array:
        act = self.activation_fn(net)
        return mx.add(prev, mx.multiply(self.ode_unfolds, act))

    def _rk4(self, prev: mx.array, net: mx.array) -> mx.array:
        dt = mx.divide(1.0, self.ode_unfolds)
        f = self.activation_fn
        k1 = f(net)
        k2 = f(mx.add(net, mx.multiply(mx.multiply(0.5, dt), k1)))
        k3 = f(mx.add(net, mx.multiply(mx.multiply(0.5, dt), k2)))
        k4 = f(mx.add(net, mx.multiply(dt, k3)))
        return mx.add(prev, mx.multiply(mx.divide(dt, 6.0), mx.add(mx.add(k1, mx.multiply(2.0, k2)), mx.add(mx.multiply(2.0, k3), k4))))

    def __call__(
        self,
        inputs: mx.array,
        state: Optional[mx.array] = None,
        time: float | mx.array = 1.0,
    ) -> tuple[mx.array, mx.array]:
        del time  # original implementation ignores elapsed time
        batch = inputs.shape[0]
        self._ensure_params(inputs.shape[-1])
        if state is None:
            state = mx.zeros((batch, self.units), dtype=mx.float32)

        net = self._net_input(inputs, state)
        if self.solver == "semi_implicit":
            new_state = self._semi_implicit(state, net)
        elif self.solver == "explicit":
            new_state = self._explicit(state, net)
        else:
            new_state = self._rk4(state, net)
        return new_state, new_state
