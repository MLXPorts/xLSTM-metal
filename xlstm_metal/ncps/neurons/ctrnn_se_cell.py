"""MLX CTRNN mirroring LTC-SE's TensorFlow implementation.

Based on Bidollahkhani et al., 2023 (Apache-2.0, LTC-SE repository).
"""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .hyperprofiles import HyperProfile, load_profile


class CTRNNSECell(nn.Module):
    def __init__(
        self,
        units: int,
        profile: Optional[Union[str, HyperProfile]] = None,
    ) -> None:
        super().__init__()
        self.units = units
        if profile is None:
            self._profile = load_profile("ctrnn_tf")
        elif isinstance(profile, str):
            self._profile = load_profile(profile)
        else:
            self._profile = profile
        extras = self._profile.extras

        self._unfolds = int(self._profile.ode_unfolds or extras.get("unfolds", 6))
        self._delta_t = float(extras.get("delta_t", 0.1))
        self._cell_clip = float(extras.get("cell_clip", -1.0))
        self._global_feedback = bool(extras.get("global_feedback", False))
        self._fix_tau = bool(extras.get("fix_tau", True))
        self._tau_value = float(extras.get("tau", 1.0))

        self.linear_step: Optional[nn.Linear] = None
        self._input_dim: Optional[int] = None
        self.tau_param: Optional[mx.array] = None

    def ensure_parameters(self, input_dim: int) -> None:
        if self._input_dim == input_dim and self.linear_step is not None:
            return

        in_features = input_dim + self.units if self._global_feedback else input_dim
        self.linear_step = nn.Linear(in_features, self.units, bias=True)
        self.linear_step.bias = mx.zeros((self.units,), dtype=mx.float32)
        self._input_dim = input_dim

        if not self._fix_tau and self.tau_param is None:
            self.tau_param = mx.array([self._tau_value], dtype=mx.float32)

    def _compute_tau(self) -> mx.array:
        if self._fix_tau:
            return mx.array([self._tau_value], dtype=mx.float32)
        softplus = mx.log1p(mx.exp(self.tau_param))
        return softplus

    def __call__(self, inputs: mx.array, state: Optional[mx.array] = None) -> tuple[mx.array, mx.array]:
        self.ensure_parameters(inputs.shape[-1])

        if state is None:
            state = mx.zeros((inputs.shape[0], self.units), dtype=mx.float32)

        tau = self._compute_tau()

        for _ in range(self._unfolds):
            if self._global_feedback:
                step_input = mx.concatenate([inputs, state], axis=1)
            else:
                step_input = inputs

            activation = mx.tanh(self.linear_step(step_input))
            f_prime = -state / tau + activation
            state = state + self._delta_t * f_prime

            if self._cell_clip > 0:
                state = mx.clip(state, -self._cell_clip, self._cell_clip)

        return state, state
