#!/usr/bin/env python
"""
MLX activation registry with Keras-like string → callable mapping.

Goals
- Provide stable, named activations with consistent constants/dtypes.
- Allow backend-compensation (Torch-like vs MLX-stable) via names.
- Permit custom injections (e.g., bit-exact or kernel-backed functions).

Usage
- get_activation(name: str, **kwargs) -> Callable[mx.array] -> mx.array
- register_activation(name: str, fn: Callable)
"""

from typing import Callable, Dict, Optional

import mlx.core as mx
try:
    from ..math_ops import sqrt_2_over_pi  # type: ignore
except Exception:
    sqrt_2_over_pi = None


# ---- Built-in primitives (float32 throughout) ----

def _tanh(x: mx.array) -> mx.array:
    return mx.tanh(x)


def _lecun_tanh(x: mx.array, scale: float = 1.7159, slope: float = 0.666) -> mx.array:
    # Ensure float32 constants and MLX ops
    s = mx.array(scale, dtype=mx.float32)
    k = mx.array(slope, dtype=mx.float32)
    return mx.multiply(s, mx.tanh(mx.multiply(k, x)))


def _sigmoid(x: mx.array) -> mx.array:
    # MLX fused version
    return mx.sigmoid(x)


def _relu(x: mx.array) -> mx.array:
    zero = mx.array(0.0, dtype=x.dtype)
    return mx.maximum(x, zero)


def _silu(x: mx.array) -> mx.array:
    return mx.multiply(x, mx.sigmoid(x))


def _gelu_erf(x: mx.array) -> mx.array:
    # Exact GELU via erf (matches Torch approximate='none') using MLX ops only
    one = mx.array(1.0, dtype=mx.float32)
    half = mx.array(0.5, dtype=mx.float32)
    two = mx.array(2.0, dtype=mx.float32)
    c = mx.sqrt(two)
    return mx.multiply(half, mx.multiply(x, mx.add(one, mx.erf(mx.divide(x, c)))))


def _gelu_tanh(x: mx.array) -> mx.array:
    # Tanh approx (matches Torch approximate='tanh') using MLX ops only
    half = mx.array(0.5, dtype=mx.float32)
    one = mx.array(1.0, dtype=mx.float32)
    c044715 = mx.array(0.044715, dtype=mx.float32)
    if sqrt_2_over_pi is not None:
        c = sqrt_2_over_pi() if callable(sqrt_2_over_pi) else sqrt_2_over_pi
    else:
        c = mx.sqrt(mx.divide(mx.array(2.0, dtype=mx.float32), mx.array(3.141592653589793, dtype=mx.float32)))
    # x^3 using MLX power
    x3 = mx.power(x, mx.array(3.0, dtype=x.dtype))
    inner = mx.add(x, mx.multiply(c044715, x3))
    tanh_arg = mx.multiply(c, inner)
    return mx.multiply(half, mx.multiply(x, mx.add(one, mx.tanh(tanh_arg))))


def _identity(x: mx.array) -> mx.array:
    return x


# ---- Registry ----

_REGISTRY: Dict[str, Callable[[mx.array], mx.array]] = {
    "tanh": _tanh,
    "lecun_tanh": _lecun_tanh,
    "sigmoid": _sigmoid,
    "relu": _relu,
    "silu": _silu,
    "gelu": _gelu_erf,        # alias to exact
    "gelu_erf": _gelu_erf,
    "gelu_tanh": _gelu_tanh,
    "identity": _identity,
}


def register_activation(name: str, fn: Callable[[mx.array], mx.array]) -> None:
    """Register a custom activation under a given name."""
    _REGISTRY[name] = fn


def get_activation(name: str, *, scale: Optional[float] = None, slope: Optional[float] = None) -> Callable[[mx.array], mx.array]:
    """Return a callable activation by name.

    Supports optional tuning for LeCun tanh:
      get_activation('lecun_tanh', scale=1.7159, slope=0.666)
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(_REGISTRY.keys())}")

    fn = _REGISTRY[key]
    if key == 'lecun_tanh' and (scale is not None or slope is not None):
        sc = 1.7159 if scale is None else scale
        sl = 0.666 if slope is None else slope
        def _lecun_bound(x: mx.array) -> mx.array:
            return _lecun_tanh(x, sc, sl)
        return _lecun_bound
    return fn
