"""MLX port of the TensorFlow LTCCell from experiments_with_ltcs/ltc_model.py."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .hyperprofiles import HyperProfile, load_profile


class LTCSECell(nn.Module):
    """MLX implementation of the LTC-SE LTCCell (Bidollahkhani et al., 2023).

    This derivative work reimplements the TensorFlow LTCCell published in
    the LTC-SE repository (Apache-2.0, Hasani & Bidollahkhani et al.).
    """

    def __init__(
        self,
        units: int,
        *,
        profile: Optional[HyperProfile] = None,
        erev_init_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.units = units
        self._profile = profile or load_profile("ltcse_tf")
        self._ode_unfolds = int(self._profile.ode_unfolds or 6)
        self._solver = self._profile.solver or "semi_implicit"
        self._input_mapping = self._profile.input_mapping or "affine"
        self._erev_init_factor = erev_init_factor

        # Initialisation ranges match ltc_model.LTCCell
        init_cfg = self._profile.initializers
        self._w_init_min = init_cfg.get("w_min", 0.01)
        self._w_init_max = init_cfg.get("w_max", 1.0)
        self._cm_init_min = init_cfg.get("cm_min", 0.5)
        self._cm_init_max = init_cfg.get("cm_max", 0.5)
        self._gleak_init_min = init_cfg.get("gleak_min", 1.0)
        self._gleak_init_max = init_cfg.get("gleak_max", 1.0)
        self._mu_init = (init_cfg.get("mu_min", 0.3), init_cfg.get("mu_max", 0.8))
        self._sigma_init = (
            init_cfg.get("sigma_min", 3.0),
            init_cfg.get("sigma_max", 8.0),
        )
        self._vleak_init = (
            init_cfg.get("vleak_min", -0.2),
            init_cfg.get("vleak_max", 0.2),
        )

        clamp_cfg = self._profile.constraints
        self._w_min_value = clamp_cfg.get("w_min", 1e-5)
        self._w_max_value = clamp_cfg.get("w_max", 1e3)
        self._gleak_min_value = clamp_cfg.get("gleak_min", 1e-5)
        self._gleak_max_value = clamp_cfg.get("gleak_max", 1e3)
        self._cm_t_min_value = clamp_cfg.get("cm_min", 1e-6)
        self._cm_t_max_value = clamp_cfg.get("cm_max", 1e3)

        self._input_dim: Optional[int] = None

    def _ensure_params(self, input_dim: int) -> None:
        if self._input_dim is not None:
            return
        self._input_dim = input_dim

        self.sensory_mu = mx.random.uniform(
            low=self._mu_init[0],
            high=self._mu_init[1],
            shape=(input_dim, self.units),
            dtype=mx.float32,
        )
        self.sensory_sigma = mx.random.uniform(
            low=self._sigma_init[0],
            high=self._sigma_init[1],
            shape=(input_dim, self.units),
            dtype=mx.float32,
        )
        self.sensory_W = mx.random.uniform(
            low=self._w_init_min,
            high=self._w_init_max,
            shape=(input_dim, self.units),
            dtype=mx.float32,
        )
        sensory_sign = mx.random.randint(0, 2, shape=(input_dim, self.units))
        sensory_sign = mx.subtract(mx.multiply(sensory_sign, 2), 1)
        self.sensory_erev = mx.multiply(
            sensory_sign.astype(mx.float32), self._erev_init_factor
        )

        self.mu = mx.random.uniform(
            low=self._mu_init[0],
            high=self._mu_init[1],
            shape=(self.units, self.units),
            dtype=mx.float32,
        )
        self.sigma = mx.random.uniform(
            low=self._sigma_init[0],
            high=self._sigma_init[1],
            shape=(self.units, self.units),
            dtype=mx.float32,
        )
        self.W = mx.random.uniform(
            low=self._w_init_min,
            high=self._w_init_max,
            shape=(self.units, self.units),
            dtype=mx.float32,
        )
        erev_sign = mx.random.randint(0, 2, shape=(self.units, self.units))
        erev_sign = mx.subtract(mx.multiply(erev_sign, 2), 1)
        self.erev = mx.multiply(erev_sign.astype(mx.float32), self._erev_init_factor)

        self.vleak = mx.random.uniform(
            low=self._vleak_init[0],
            high=self._vleak_init[1],
            shape=(self.units,),
            dtype=mx.float32,
        )
        self.gleak = mx.full((self.units,), self._gleak_init_min, dtype=mx.float32)
        self.cm_t = mx.full((self.units,), self._cm_init_min, dtype=mx.float32)

        if self._input_mapping in {"affine", "linear"}:
            self.input_w = mx.ones((input_dim,), dtype=mx.float32)
        else:
            self.input_w = None
        if self._input_mapping == "affine":
            self.input_b = mx.zeros((input_dim,), dtype=mx.float32)
        else:
            self.input_b = None

    def _map_inputs(self, inputs: mx.array) -> mx.array:
        result = inputs
        if self.input_w is not None:
            result = mx.multiply(result, self.input_w)
        if self.input_b is not None:
            result = mx.add(result, self.input_b)
        return result

    def _sigmoid(self, potentials: mx.array, mu: mx.array, sigma: mx.array) -> mx.array:
        potentials = mx.expand_dims(potentials, axis=2)
        mu_exp = mx.expand_dims(mu, axis=0)
        sigma_exp = mx.expand_dims(sigma, axis=0)
        x = mx.multiply(sigma_exp, mx.subtract(potentials, mu_exp))
        return mx.divide(1.0, mx.add(1.0, mx.exp(mx.negative(x))))

    def _sensory_activation(self, inputs: mx.array) -> mx.array:
        sig = self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        W = mx.expand_dims(self.sensory_W, axis=0)
        return mx.multiply(W, sig)

    def _recurrent_activation(self, state: mx.array) -> mx.array:
        sig = self._sigmoid(state, self.mu, self.sigma)
        W = mx.expand_dims(self.W, axis=0)
        return mx.multiply(W, sig)

    def _ode_step(self, inputs: mx.array, state: mx.array) -> mx.array:
        v_pre = state
        sensory_w_activation = self._sensory_activation(inputs)
        sensory_rev_activation = mx.multiply(
            sensory_w_activation, mx.expand_dims(self.sensory_erev, axis=0)
        )

        w_num_sensory = mx.sum(sensory_rev_activation, axis=1)
        w_den_sensory = mx.sum(sensory_w_activation, axis=1)

        cm = mx.expand_dims(self.cm_t, axis=0)
        gleak = mx.expand_dims(self.gleak, axis=0)
        vleak = mx.expand_dims(self.vleak, axis=0)

        for _ in range(self._ode_unfolds):
            w_activation = self._recurrent_activation(v_pre)
            rev_activation = mx.multiply(
                w_activation, mx.expand_dims(self.erev, axis=0)
            )

            w_numerator = mx.add(mx.sum(rev_activation, axis=1), w_num_sensory)
            w_denominator = mx.add(mx.sum(w_activation, axis=1), w_den_sensory)

            numerator = mx.add(
                mx.add(mx.multiply(cm, v_pre), mx.multiply(gleak, vleak)), w_numerator
            )
            denominator = mx.add(mx.add(cm, gleak), w_denominator)
            v_pre = mx.divide(numerator, denominator)

        return v_pre

    def _ode_step_explicit(self, inputs: mx.array, state: mx.array) -> mx.array:
        v_pre = state
        sensory_w_activation = self._sensory_activation(inputs)
        w_reduced_sensory = mx.sum(sensory_w_activation, axis=1)

        for _ in range(self._ode_unfolds):
            w_activation = self._recurrent_activation(v_pre)
            w_reduced_synapse = mx.sum(w_activation, axis=1)

            sensory_in = mx.multiply(
                mx.expand_dims(self.sensory_erev, axis=0), sensory_w_activation
            )
            synapse_in = mx.multiply(mx.expand_dims(self.erev, axis=0), w_activation)

            sum_in = mx.add(
                mx.subtract(
                    mx.add(
                        mx.sum(sensory_in, axis=1),
                        mx.subtract(
                            mx.sum(synapse_in, axis=1),
                            mx.multiply(v_pre, w_reduced_synapse),
                        ),
                    ),
                    mx.multiply(v_pre, w_reduced_sensory),
                ),
                0.0,
            )

            f_prime = mx.multiply(
                mx.divide(1.0, self.cm_t),
                mx.add(
                    mx.multiply(self.gleak, mx.subtract(self.vleak, v_pre)), sum_in
                ),
            )
            v_pre = mx.add(v_pre, mx.multiply(0.1, f_prime))

        return v_pre

    def _f_prime(self, inputs: mx.array, state: mx.array) -> mx.array:
        sensory_w_activation = self._sensory_activation(inputs)
        w_reduced_sensory = mx.sum(sensory_w_activation, axis=1)

        w_activation = self._recurrent_activation(state)
        w_reduced_synapse = mx.sum(w_activation, axis=1)

        sensory_in = mx.multiply(
            mx.expand_dims(self.sensory_erev, axis=0), sensory_w_activation
        )
        synapse_in = mx.multiply(mx.expand_dims(self.erev, axis=0), w_activation)

        sum_in = mx.add(
            mx.subtract(
                mx.add(
                    mx.sum(sensory_in, axis=1),
                    mx.subtract(
                        mx.sum(synapse_in, axis=1),
                        mx.multiply(state, w_reduced_synapse),
                    ),
                ),
                mx.multiply(state, w_reduced_sensory),
            ),
            0.0,
        )

        return mx.multiply(
            mx.divide(1.0, self.cm_t),
            mx.add(mx.multiply(self.gleak, mx.subtract(self.vleak, state)), sum_in),
        )

    def _ode_step_runge_kutta(self, inputs: mx.array, state: mx.array) -> mx.array:
        h = 0.1
        v = state
        for _ in range(self._ode_unfolds):
            k1 = mx.multiply(h, self._f_prime(inputs, v))
            k2 = mx.multiply(h, self._f_prime(inputs, mx.add(v, mx.multiply(0.5, k1))))
            k3 = mx.multiply(h, self._f_prime(inputs, mx.add(v, mx.multiply(0.5, k2))))
            k4 = mx.multiply(h, self._f_prime(inputs, mx.add(v, k3)))
            v = mx.add(
                v,
                mx.divide(
                    mx.add(
                        mx.add(k1, mx.multiply(2, k2)),
                        mx.add(mx.multiply(2, k3), k4),
                    ),
                    6.0,
                ),
            )
        return v

    def __call__(
        self,
        inputs: mx.array,
        state: Optional[mx.array] = None,
        elapsed_time: float | mx.array = 1.0,
    ) -> tuple[mx.array, mx.array]:
        if inputs.ndim != 2:
            raise ValueError("Legacy LTCCell expects [batch, features] inputs")
        self._ensure_params(inputs.shape[-1])

        inputs = inputs.astype(mx.float32)
        mapped_inputs = self._map_inputs(inputs)

        if state is None:
            batch = inputs.shape[0]
            state = mx.zeros((batch, self.units), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)

        if self._solver == "explicit":
            next_state = self._ode_step_explicit(mapped_inputs, state)
        elif self._solver == "rk4":
            next_state = self._ode_step_runge_kutta(mapped_inputs, state)
        else:
            next_state = self._ode_step(mapped_inputs, state)

        return next_state, next_state

    def apply_constraints(self) -> None:
        self.cm_t = mx.clip(self.cm_t, self._cm_t_min_value, self._cm_t_max_value)
        self.gleak = mx.clip(self.gleak, self._gleak_min_value, self._gleak_max_value)
        self.W = mx.clip(self.W, self._w_min_value, self._w_max_value)
        self.sensory_W = mx.clip(
            self.sensory_W, self._w_min_value, self._w_max_value
        )
