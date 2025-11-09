"""MLX Neural-ODE cell derived from the LTC-SE TensorFlow code.

Attribution: Bidollahkhani et al., 2023 (Apache-2.0).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .hyperprofiles import HyperProfile, load_profile


class NODESECell(nn.Module):
    def __init__(
        self,
        units: int,
        profile: Optional[HyperProfile] = None,
    ) -> None:
        super().__init__()
        self.units = units
        self._profile = profile or load_profile("node_tf")
        extras = self._profile.extras

        self._unfolds = int(self._profile.ode_unfolds or extras.get("unfolds", 6))
        self._delta_t = float(extras.get("delta_t", 0.1))
        self._cell_clip = float(extras.get("cell_clip", -1.0))

        self.linear_step: Optional[nn.Linear] = None
        self._input_dim: Optional[int] = None

    def _ensure_parameters(self, input_dim: int) -> None:
        if self._input_dim == input_dim and self.linear_step is not None:
            return

        in_features = input_dim + self.units  # Python int ops OK
        self.linear_step = nn.Linear(in_features, self.units, bias=True)
        self.linear_step.bias = mx.zeros((self.units,), dtype=mx.float32)
        self._input_dim = input_dim

    def _f_prime(self, inputs: mx.array, state: mx.array) -> mx.array:
        fused = mx.concatenate([inputs, state], axis=1)
        return mx.tanh(self.linear_step(fused))

    def __call__(self, inputs: mx.array, state: Optional[mx.array] = None) -> tuple[mx.array, mx.array]:
        self._ensure_parameters(inputs.shape[-1])

        if state is None:
            state = mx.zeros((inputs.shape[0], self.units), dtype=mx.float32)

        for _ in range(self._unfolds):
            k1 = mx.multiply(self._delta_t, self._f_prime(inputs, state))
            k2 = mx.multiply(
                self._delta_t, self._f_prime(inputs, mx.add(state, mx.multiply(0.5, k1)))
            )
            k3 = mx.multiply(
                self._delta_t, self._f_prime(inputs, mx.add(state, mx.multiply(0.5, k2)))
            )
            k4 = mx.multiply(self._delta_t, self._f_prime(inputs, mx.add(state, k3)))
            state = mx.add(
                state,
                mx.divide(
                    mx.add(
                        mx.add(k1, mx.multiply(2, k2)),
                        mx.add(mx.multiply(2, k3), k4),
                    ),
                    6.0,
                ),
            )

            if self._cell_clip > 0:
                state = mx.clip(state, -self._cell_clip, self._cell_clip)

        return state, state
