from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_KERNEL = r"""
    uint i = thread_position_in_grid.x;
    int size = int(shape[0]);
    if (i >= size) return;
    // Read input and scalar cap (as a single-element buffer)
    float x = inp[i];
    float c = cap[0];
    // Clamp to +/- c using tanh-based cap (or min/max form as needed)
    out[i] = c * tanh(x / c);
    """


class SoftCapMetalKernel:
    """Stateless Metal kernel wrapper for the soft-cap primitive."""

    def __init__(self) -> None:
        self._kernel: Optional[mx.fast.metal_kernel] = None

    def compile(self):
        """
        Compile the Metal kernel and return it.

        This can be called early on a global level to pre-compile the shader
        instead of compiling on first use.

        Returns:
            mx.fast.metal_kernel: The compiled Metal kernel.
        """
        if self._kernel is None:
            self._kernel = mx.fast.metal_kernel(
                name="soft_cap",
                input_names=["inp", "cap", "shape"],
                output_names=["out"],
                header=_HEADER,
                source=_KERNEL,
            )
        return self._kernel

    def apply(self, x: mx.array, cap_tensor: mx.array) -> mx.array:
        """Apply the soft-cap kernel to ``x`` using ``cap_tensor``."""
        kernel = self.compile()
        x_arr = mx.array(x, dtype=mx.float32)
        x_flat = x_arr.reshape(-1)
        shape_arr = mx.array([x_flat.size], dtype=mx.uint32)

        grid = (x_flat.size, 1, 1)
        threadgroup = (256, 1, 1)

        (out_flat,) = kernel(
            inputs=[x_flat, cap_tensor.reshape(1), shape_arr],
            output_shapes=[x_flat.shape],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
        )
        return out_flat.reshape(x_arr.shape)


class SoftCapCell(nn.Module):
    """NCPS-style cell that applies the Metal soft-cap kernel."""

    def __init__(self, cap_value: Optional[float] = None, kernel: Optional[SoftCapMetalKernel] = None) -> None:
        super().__init__()
        self.default_cap = cap_value
        self.kernel = kernel or SoftCapMetalKernel()

    def __call__(self, x: mx.array, cap_value: Optional[mx.array | float] = None) -> mx.array:
        cap = cap_value if cap_value is not None else self.default_cap
        if cap is None:
            return x

        cap_tensor = mx.array(cap, dtype=mx.float32)
        zero = mx.zeros_like(cap_tensor)
        if bool(mx.less_equal(cap_tensor, zero).item()):
            raise ValueError("cap_value must be positive")

        return self.kernel.apply(x, cap_tensor)


# Functional interface for convenience
soft_cap = SoftCapCell()

__all__ = ['soft_cap', 'SoftCapCell', 'SoftCapMetalKernel']
