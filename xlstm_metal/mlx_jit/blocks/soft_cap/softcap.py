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
    """nn.Module wrapper around the Metal soft-cap kernel."""

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
    """NCPS-style cell that applies the Metal soft-cap kernel."""

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
