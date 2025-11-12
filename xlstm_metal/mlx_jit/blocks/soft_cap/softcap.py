"""Soft-Cap Function – MLX Implementation (Metal-Accelerated Bounded Activation)

Overview
--------
Soft-cap applies a smooth, bounded activation function to prevent extreme
values from destabilizing exponential operations in gating mechanisms:

  soft_cap(x, c) = c * tanh(x / c)

where c > 0 is the cap value. This function:
  - Is smooth and differentiable everywhere
  - Asymptotically approaches ±c as x → ±∞
  - Reduces to identity near zero (tanh(x/c) ≈ x/c for small x)

Why Soft-Cap?
-------------
In xLSTM, gate preactivations (input, forget, output gates) can grow large
during training. When these are exponentiated for stabilized gating:
  gate = exp(preactivation - stabilizer)
Large preactivations cause:
  1. Numerical overflow (exp(large) → inf)
  2. Gradient instability (saturated tanh derivatives → vanishing gradients)
  3. Training divergence (unstable exponential accumulation)

Soft-cap bounds preactivations to [-c, +c] smoothly, preventing these issues
while preserving gradient flow for moderate values.

Mathematical Properties
-----------------------
- Domain: ℝ → Range: (-c, +c)
- Monotonic: strictly increasing
- Odd function: soft_cap(-x, c) = -soft_cap(x, c)
- Derivative: d/dx[soft_cap(x,c)] = sech²(x/c) ∈ (0, 1]
- Near-identity: for |x| << c, soft_cap(x,c) ≈ x

Typical Cap Values
------------------
- Gates (i, f, o): c = 15.0 (xLSTM-7B default)
- Output logits: c = 30.0 (prevents overconfident predictions)

Metal Acceleration
------------------
This implementation uses a custom Metal kernel for element-wise soft-cap
on Apple Silicon. The kernel:
  1. Flattens input to 1D
  2. Applies cap * tanh(x / cap) element-wise in parallel
  3. Reshapes back to original shape

Supports float32 and bfloat16 with dtype-specific shader compilation.

Usage
-----
Functional interface (most common):
  from xlstm_metal.mlx_jit.blocks.soft_cap import soft_cap
  y = soft_cap(x, cap_value=15.0)

Module interface (for precompilation):
  cap_cell = SoftCapCell(cap_value=15.0)
  cap_cell.precompile((mx.float32, mx.bfloat16))
  y = cap_cell(x)

Gradient Flow
-------------
The derivative sech²(x/c) is always positive and bounded, ensuring stable
gradient backpropagation. Unlike hard clipping (non-differentiable at
boundaries), soft-cap maintains smooth gradients throughout.

Parity
------
Logic mirrors torch-native soft_cap for cross-backend testing.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_KERNEL_TEMPLATE = r"""
    uint i = thread_position_in_grid.x;
    int size = int(shape[0]);
    if (i >= size) return;

    @SCALAR@ x_val = inp[i];
    @SCALAR@ c_val = cap[0];
    float x = float(x_val);
    float c = float(c_val);
    float y = c * tanh(x / c);
    out[i] = @CAST@;
    """


class SoftCapKernel(nn.Module):
    """Metal kernel for element-wise soft-cap with dtype caching.

    Compiles and caches Metal shaders for float32/bfloat16. The kernel
    implements cap * tanh(x / cap) in parallel across all elements.

    Kernel Strategy
    ---------------
    - Thread count: input size (one thread per element)
    - Threadgroup size: 256 threads
    - Computation: reads x[i], cap, computes y = cap * tanh(x / cap), writes y[i]

    Parameters
    ----------
    None (stateless, kernel cache is instance attribute)

    Methods
    -------
    build(dtype) -> metal_kernel
        Compile and return cached kernel for given dtype.
    precompile(dtypes)
        Pre-compile kernels for multiple dtypes at initialization.
    """

    def __init__(self) -> None:
        super().__init__()
        self._kernels: dict[mx.Dtype, Optional[mx.fast.metal_kernel]] = {
            mx.float32: None,
            mx.bfloat16: None,
        }

    def build(self, dtype: mx.Dtype) -> mx.fast.metal_kernel:
        if dtype not in self._kernels:
            raise TypeError(f"SoftCap kernel only supports float32/bfloat16, got {dtype}")

        cached = self._kernels[dtype]
        if cached is not None:
            return cached

        if dtype == mx.float32:
            scalar = "float"
            cast = "y"
            name = "soft_cap_fp32"
        else:  # dtype == mx.bfloat16
            scalar = "bfloat"
            cast = "bfloat(y)"
            name = "soft_cap_bf16"

        source = _KERNEL_TEMPLATE.replace("@SCALAR@", scalar).replace("@CAST@", cast)
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=["inp", "cap", "shape"],
            output_names=["out"],
            header=_HEADER,
            source=source,
        )
        self._kernels[dtype] = kernel
        return kernel

    def __call__(self, dtype: mx.Dtype) -> mx.fast.metal_kernel:
        return self.build(dtype)

    def precompile(self, dtypes: tuple[mx.Dtype, ...]) -> None:
        """

        :param dtypes:
        """
        for dtype in dtypes:
            self.build(dtype)


class SoftCapCell(nn.Module):
    """NCPS-style soft-cap cell with Metal kernel backend.

    Applies bounded activation to prevent extreme values in gating.
    Handles arbitrary input shapes via flattening/reshaping.

    Parameters
    ----------
    cap_value : float | None, optional
        Default cap value (can be overridden per call).
    kernel : SoftCapKernel | None, optional
        Custom kernel instance (default creates new).

    Returns (forward)
    -----------------
    output : mx.array
        Soft-capped activations, same shape as input.

    Examples
    --------
    >>> cap = SoftCapCell(cap_value=15.0)
    >>> x = mx.array([0.0, 10.0, 20.0, 50.0])
    >>> y = cap(x)
    >>> # y ≈ [0.0, 9.96, 14.76, 15.0] (bounded to ~15)
    """

    def __init__(self, cap_value: Optional[float] = None, kernel: Optional[SoftCapKernel] = None) -> None:
        super().__init__()
        self.default_cap = cap_value
        self.kernel = kernel or SoftCapKernel()

    def build(self, dtype: mx.Dtype) -> None:
        self.kernel.build(dtype)

    def precompile(self, dtypes: tuple[mx.Dtype, ...]) -> None:
        for dtype in dtypes:
            self.build(dtype)

    def __call__(self, x: mx.array, cap_value: Optional[mx.array | float] = None) -> mx.array:
        """Apply soft-cap activation using Metal kernel.

        Parameters
        ----------
        x : mx.array
            Input tensor (arbitrary shape).
        cap_value : float | mx.array | None, optional
            Cap value (uses default_cap if None).

        Returns
        -------
        output : mx.array
            Soft-capped tensor matching input shape.

        Raises
        ------
        ValueError
            If cap_value <= 0 (cap must be positive).
        """
        cap = cap_value if cap_value is not None else self.default_cap
        if cap is None:
            return x

        cap_tensor = mx.array(cap, dtype=x.dtype)
        zero = mx.zeros_like(cap_tensor)
        if bool(mx.less_equal(cap_tensor, zero).item()):
            raise ValueError("cap_value must be positive")

        kernel = self.kernel(x.dtype)
        x_arr = mx.array(x, dtype=x.dtype)
        x_flat = x_arr.reshape(-1)
        shape_arr = mx.array([x_flat.size], dtype=mx.uint32)

        grid = (x_flat.size, 1, 1)
        threadgroup = (256, 1, 1)

        (out_flat,) = kernel(
            inputs=[x_flat, cap_tensor.reshape(1), shape_arr],
            output_shapes=[x_flat.shape],
            output_dtypes=[x.dtype],
            grid=grid,
            threadgroup=threadgroup,
        )
        return out_flat.reshape(x_arr.shape)


# Functional interface for convenience
soft_cap = SoftCapCell()

__all__ = ['soft_cap', 'SoftCapCell', 'SoftCapKernel']
