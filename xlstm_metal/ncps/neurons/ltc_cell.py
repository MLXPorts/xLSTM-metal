"""MLX implementation of the Liquid Time-Constant (LTC) cell."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class LTCCell(nn.Module):
    """Liquid time-constant (LTC) cell mirroring the original Torch implementation."""

    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
    ) -> None:
        super().__init__()

        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown input dimension. Provide 'in_features' or call wiring.build()."
            )

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = int(ode_unfolds)
        self._epsilon = epsilon
        self._implicit_param_constraints = implicit_param_constraints

        self._positive = nn.Softplus() if implicit_param_constraints else None
        self._clip = lambda arr: mx.maximum(arr, mx.zeros_like(arr))

        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3.0, 8.0),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3.0, 8.0),
            "sensory_mu": (0.3, 0.8),
        }

        self._allocate_parameters()

        # Store sparsity masks / reversal potentials as non-trainable buffers
        self._sparsity_mask = mx.array(mx.abs(wiring.adjacency_matrix), dtype=mx.float32)
        self._sensory_sparsity_mask = (
            mx.array(mx.abs(wiring.sensory_adjacency_matrix), dtype=mx.float32)
            if wiring.sensory_adjacency_matrix is not None
            else None
        )
        self._erev = mx.array(wiring.erev_initializer(), dtype=mx.float32)
        self._sensory_erev = (
            mx.array(wiring.sensory_erev_initializer(), dtype=mx.float32)
            if wiring.sensory_adjacency_matrix is not None
            else None
        )

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def state_size(self) -> int:
        return self._wiring.units

    @property
    def sensory_size(self) -> int:
        return self._wiring.input_dim

    @property
    def motor_size(self) -> int:
        return self._wiring.output_dim

    @property
    def output_size(self) -> int:
        return self.motor_size

    # --------------------------------------------------------------------- #
    # Parameter allocation helpers
    # --------------------------------------------------------------------- #
    def _random_uniform(self, shape, minimum: float, maximum: float) -> mx.array:
        if minimum == maximum:
            return mx.full(shape, minimum, dtype=mx.float32)
        return mx.random.uniform(
            low=minimum, high=maximum, shape=shape, dtype=mx.float32
        )

    def _init_tensor(self, shape, name: str) -> mx.array:
        minval, maxval = self._init_ranges[name]
        return self._random_uniform(shape, minval, maxval)

    def _allocate_parameters(self) -> None:
        units = self.state_size
        sensory = self.sensory_size

        self.gleak = self._init_tensor((units,), "gleak")
        self.vleak = self._init_tensor((units,), "vleak")
        self.cm = self._init_tensor((units,), "cm")

        self.sigma = self._init_tensor((units, units), "sigma")
        self.mu = self._init_tensor((units, units), "mu")
        self.w = self._init_tensor((units, units), "w")

        self.sensory_sigma = self._init_tensor((sensory, units), "sensory_sigma")
        self.sensory_mu = self._init_tensor((sensory, units), "sensory_mu")
        self.sensory_w = self._init_tensor((sensory, units), "sensory_w")

        if self._input_mapping in ("affine", "linear"):
            self.input_w = mx.ones((sensory,), dtype=mx.float32)
        if self._input_mapping == "affine":
            self.input_b = mx.zeros((sensory,), dtype=mx.float32)

        if self._output_mapping in ("affine", "linear"):
            self.output_w = mx.ones((self.motor_size,), dtype=mx.float32)
        if self._output_mapping == "affine":
            self.output_b = mx.zeros((self.motor_size,), dtype=mx.float32)

    # --------------------------------------------------------------------- #
    # Core math helpers
    # --------------------------------------------------------------------- #
    def _make_positive(self, value: mx.array) -> mx.array:
        if self._positive is None:
            return value
        return self._positive(value)

    def _sigmoid(self, potentials: mx.array, mu: mx.array, sigma: mx.array) -> mx.array:
        potentials = mx.expand_dims(potentials, axis=-1)
        x = mx.multiply(sigma, mx.subtract(potentials, mu))
        return mx.divide(1.0, mx.add(1.0, mx.exp(mx.negative(x))))

    def _map_inputs(self, inputs: mx.array) -> mx.array:
        mapped = inputs
        if self._input_mapping in ("affine", "linear"):
            mapped = mx.multiply(mapped, self.input_w)
        if self._input_mapping == "affine":
            mapped = mx.add(mapped, self.input_b)
        return mapped

    def _map_outputs(self, state: mx.array) -> mx.array:
        output = state
        if self.motor_size < self.state_size:
            output = output[:, : self.motor_size]
        if self._output_mapping in ("affine", "linear"):
            output = mx.multiply(output, self.output_w)
        if self._output_mapping == "affine":
            output = mx.add(output, self.output_b)
        return output

    def apply_weight_constraints(self) -> None:
        if self._implicit_param_constraints:
            return
        self.w = self._clip(self.w)
        self.sensory_w = self._clip(self.sensory_w)
        self.cm = self._clip(self.cm)
        self.gleak = self._clip(self.gleak)

    # --------------------------------------------------------------------- #
    # Solver
    # --------------------------------------------------------------------- #
    def _ode_solver(self, inputs: mx.array, state: mx.array, elapsed_time) -> mx.array:
        v_pre = state

        if not isinstance(elapsed_time, mx.array):
            elapsed_time = mx.array(elapsed_time, dtype=mx.float32)
        if elapsed_time.ndim == 0:
            dt = mx.divide(elapsed_time, self._ode_unfolds)
            dt = mx.reshape(dt, (1, 1))
        else:
            dt = mx.reshape(mx.divide(elapsed_time, self._ode_unfolds), (-1, 1))

        sensory_activation = self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_w_activation = mx.multiply(
            self._make_positive(self.sensory_w), sensory_activation
        )
        sensory_w_activation = mx.multiply(
            sensory_w_activation, self._sensory_sparsity_mask
        )
        sensory_rev_activation = mx.multiply(sensory_w_activation, self._sensory_erev)

        w_numerator_sensory = mx.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = mx.sum(sensory_w_activation, axis=1)

        cm = mx.expand_dims(self._make_positive(self.cm), axis=0)
        cm_t = mx.divide(cm, dt)

        w_param = self._make_positive(self.w)
        w_param = mx.expand_dims(w_param, axis=0)
        gleak = mx.expand_dims(self._make_positive(self.gleak), axis=0)
        vleak = mx.expand_dims(self.vleak, axis=0)

        for _ in range(self._ode_unfolds):
            sigmoid = self._sigmoid(v_pre, self.mu, self.sigma)
            w_activation = mx.multiply(w_param, sigmoid)
            w_activation = mx.multiply(w_activation, self._sparsity_mask)
            rev_activation = mx.multiply(w_activation, self._erev)

            w_numerator = mx.add(mx.sum(rev_activation, axis=1), w_numerator_sensory)
            w_denominator = mx.add(mx.sum(w_activation, axis=1), w_denominator_sensory)

            numerator = mx.add(
                mx.add(mx.multiply(cm_t, v_pre), mx.multiply(gleak, vleak)),
                w_numerator,
            )
            denominator = mx.add(mx.add(cm_t, gleak), w_denominator)
            v_pre = mx.divide(numerator, mx.add(denominator, self._epsilon))

        return v_pre

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def __call__(
        self, inputs: mx.array, state: mx.array, elapsed_time: float | mx.array = 1.0
    ) -> tuple[mx.array, mx.array]:
        mapped_inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(mapped_inputs, state, elapsed_time)
        outputs = self._map_outputs(next_state)
        return outputs, next_state
