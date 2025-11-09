"""MLX implementation of the Closed-form Continuous-time (CfC) cell."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn


class LeCunTanh(nn.Module):
    """LeCun scaled tanh activation."""

    def __init__(self) -> None:
        super().__init__()
        self._tanh = nn.Tanh()

    def __call__(self, x: mx.array) -> mx.array:
        return mx.multiply(1.7159, self._tanh(mx.multiply(0.666, x)))


def _create_activation(name: str) -> nn.Module:
    mapping = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "lecun_tanh": LeCunTanh,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'")
    return mapping[name]()


class CfCCell(nn.Module):
    """Closed-form Continuous-time cell with optional sparsity mask."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        sparsity_mask: Optional[mx.array] = None,
        custom_activations: Optional[Dict[str, Callable[[], nn.Module]]] = None,
    ) -> None:
        super().__init__()

        allowed_modes = {"default", "pure", "no_gate"}
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {allowed_modes}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.backbone_layers = backbone_layers

        if sparsity_mask is None:
            self._sparsity_mask = None
        else:
            mask = (
                sparsity_mask
                if isinstance(sparsity_mask, mx.array)
                else mx.array(sparsity_mask)
            )
            mask = mx.abs(mask)
            mask = mx.array(mask, dtype=mx.float32)
            mask = mx.transpose(mask)
            self._sparsity_mask = mask

        cat_dim = mx.add(input_size, hidden_size)
        if backbone_layers > 0:
            modules: list[nn.Module] = []
            activation_lookup = {}
            if custom_activations:
                activation_lookup.update(custom_activations)
            modules.append(nn.Linear(cat_dim, backbone_units))
            modules.append(self._resolve_activation(activation, activation_lookup))
            if backbone_dropout > 0.0:
                modules.append(nn.Dropout(backbone_dropout))
            for _ in range(1, backbone_layers):
                modules.append(nn.Linear(backbone_units, backbone_units))
                modules.append(self._resolve_activation(activation, activation_lookup))
                if backbone_dropout > 0.0:
                    modules.append(nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*modules)
            cat_dim = backbone_units
        else:
            self.backbone = None

        self.ff1 = nn.Linear(cat_dim, hidden_size)
        if self.mode == "pure":
            self.w_tau = mx.zeros((1, hidden_size), dtype=mx.float32)
            self.A = mx.ones((1, hidden_size), dtype=mx.float32)
        else:
            self.ff2 = nn.Linear(cat_dim, hidden_size)
            self.time_a = nn.Linear(cat_dim, hidden_size)
            self.time_b = nn.Linear(cat_dim, hidden_size)

        self._sigmoid = nn.Sigmoid()

    def _resolve_activation(
        self,
        name: str,
        custom: Optional[Dict[str, Callable[[], nn.Module]]] = None,
    ) -> nn.Module:
        mapping: Dict[str, Callable[[], nn.Module]] = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "lecun_tanh": LeCunTanh,
        }
        if custom:
            mapping.update(custom)
        if name not in mapping:
            raise ValueError(
                f"Unknown activation '{name}'. Register it via 'custom_activations'."
            )
        factory = mapping[name]
        module = factory()
        if not isinstance(module, nn.Module):
            raise TypeError("Activation factory must produce an nn.Module instance.")
        return module

    def _apply_linear(self, layer: nn.Linear, x: mx.array) -> mx.array:
        weight = layer.weight
        if self._sparsity_mask is not None:
            weight = mx.multiply(weight, self._sparsity_mask)
        return mx.add(mx.matmul(x, mx.transpose(weight)), layer.bias)

    # ------------------------------------------------------------------ #
    def __call__(self, inputs: mx.array, hx: mx.array, ts: float | mx.array) -> tuple[mx.array, mx.array]:
        x = mx.concatenate([inputs, hx], axis=1)
        if self.backbone is not None:
            x = self.backbone(x)

        ff1 = self._apply_linear(self.ff1, x)
        if self.mode == "pure":
            ff1_abs = mx.abs(ff1)
            w_tau_abs = mx.abs(self.w_tau)
            if not isinstance(ts, mx.array):
                ts = mx.array(ts, dtype=mx.float32)
            ts = ts if ts.ndim > 0 else mx.reshape(ts, (1,))
            ts = mx.reshape(ts, (-1, 1))
            decay = mx.exp(mx.negative(mx.multiply(ts, mx.add(w_tau_abs, ff1_abs))))
            new_hidden = mx.add(mx.multiply(mx.negative(self.A), mx.multiply(decay, ff1)), self.A)
        else:
            ff2 = self._apply_linear(self.ff2, x)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            if not isinstance(ts, mx.array):
                ts = mx.array(ts, dtype=mx.float32)
            ts = ts if ts.ndim > 0 else mx.reshape(ts, (1,))
            ts = mx.reshape(ts, (-1, 1))
            t_interp = self._sigmoid(mx.add(mx.negative(mx.multiply(t_a, ts)), t_b))
            if self.mode == "no_gate":
                new_hidden = mx.add(ff1, mx.multiply(t_interp, ff2))
            else:
                new_hidden = mx.add(mx.multiply(ff1, mx.subtract(1.0, t_interp)), mx.multiply(t_interp, ff2))
        return new_hidden, new_hidden
