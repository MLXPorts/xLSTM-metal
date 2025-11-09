"""Enhanced Liquid Time-Constant (ELTC) cell implemented with MLX."""

from __future__ import annotations

from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from .liquid_utils import get_activation
from .ode_solvers import euler_solve, rk4_solve, semi_implicit_solve

SolverName = Literal["semi_implicit", "explicit", "rk4"]


def _masked_linear(layer: nn.Linear, mask: mx.array, x: mx.array) -> mx.array:
    """Apply a linear layer whose weights are masked by the wiring topology."""
    weight = mx.multiply(layer.weight, mask)
    return mx.add(mx.matmul(x, mx.transpose(weight)), layer.bias)


class ELTCCell(nn.Module):
    """Enhanced Liquid Time-Constant cell with configurable ODE solver."""

    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        activation: str = "tanh",
        solver: SolverName = "rk4",
        ode_unfolds: int = 4,
    ) -> None:
        super().__init__()

        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Provide 'in_features' or call wiring.build(input_dim) before creating ELTCCell."
            )

        if solver not in {"semi_implicit", "explicit", "rk4"}:
            raise ValueError("solver must be one of {'semi_implicit', 'explicit', 'rk4'}")

        self._wiring = wiring
        self._activation = get_activation(activation)
        self._solver = solver
        self._ode_unfolds = max(1, int(ode_unfolds))

        self._input_dim = wiring.input_dim
        self._state_size = wiring.units
        self._output_dim = wiring.output_dim

        self.input_linear = nn.Linear(self._input_dim, self._state_size)
        self.recurrent_linear = nn.Linear(self._state_size, self._state_size)

        sensory_mask = mx.array(mx.abs(wiring.sensory_adjacency_matrix), dtype=mx.float32)
        self._input_mask = mx.transpose(sensory_mask)

        recurrent_mask = mx.array(mx.abs(wiring.adjacency_matrix), dtype=mx.float32)
        self._recurrent_mask = mx.transpose(recurrent_mask)

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def output_size(self) -> int:
        return self._output_dim

    def _solve(self, inputs: mx.array, state: mx.array, time: float | mx.array) -> mx.array:
        input_proj = _masked_linear(self.input_linear, self._input_mask, inputs)

        def dynamics(_, y: mx.array) -> mx.array:
            recurrent_proj = _masked_linear(self.recurrent_linear, self._recurrent_mask, y)
            net = mx.add(input_proj, recurrent_proj)
            return mx.subtract(self._activation(net), y)

        if not isinstance(time, mx.array):
            time = mx.array(time, dtype=mx.float32)
        if time.ndim == 0:
            dt = mx.divide(time, self._ode_unfolds)
            dt = mx.reshape(dt, (1, 1))
        else:
            dt = mx.reshape(mx.divide(time, self._ode_unfolds), (-1, 1))

        y = state
        for _ in range(self._ode_unfolds):
            if self._solver == "semi_implicit":
                y = semi_implicit_solve(dynamics, y, dt)
            elif self._solver == "explicit":
                y = euler_solve(dynamics, y, dt)
            else:
                y = rk4_solve(dynamics, y, 0.0, dt)
        return y

    def __call__(
        self,
        inputs: mx.array,
        state: Optional[mx.array] = None,
        time: float | mx.array = 1.0,
    ) -> tuple[mx.array, mx.array]:
        batch = inputs.shape[0]
        if state is None:
            state = mx.zeros((batch, self._state_size), dtype=mx.float32)

        new_state = self._solve(inputs, state, time)
        output = new_state[:, : self._output_dim]
        return output, new_state
