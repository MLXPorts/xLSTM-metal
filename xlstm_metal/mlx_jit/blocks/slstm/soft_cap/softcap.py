import mlx.core as mx
from typing import Optional

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


class SoftCapMLXJitKernel:
    """Apply an elementwise soft cap: out = cap * tanh(x/cap).

    This class encapsulates a custom Metal kernel for the soft cap operation,
    providing a callable interface.
    """

    def __init__(self):
        """Initialize the SoftCapMLXFastKernel with no compiled kernel."""
        self.kernel: Optional[mx.fast.metal_kernel] = None

    def compile(self) -> mx.fast.metal_kernel:
        """
        Compile the Metal kernel and return it.

        This can be called early on a global level to pre-compile the shader
        instead of compiling on first use.

        Returns:
            mx.fast.metal_kernel: The compiled Metal kernel.
        """
        if self.kernel is None:
            self.kernel = mx.fast.metal_kernel(name="soft_cap", input_names=["inp", "cap", "shape"],
                                               output_names=["out"], header=_HEADER, source=_KERNEL)
        return self.kernel

    def __call__(self, x: mx.array, cap_value: float) -> mx.array:
        """
        Apply the soft cap operation.

        Parameters:
            x (mx.array): The input array.
            cap_value (float): The positive scalar cap value.

        Returns:
            mx.array: The output array with the soft cap applied.
        """
        if not isinstance(cap_value, (float, int)) or cap_value <= 0:
            raise ValueError(f"cap_value must be a positive number, but got {cap_value}")

        # Prepare inputs
        x_flat = x.reshape(-1).astype(mx.float32)
        cap_arr = mx.array([cap_value], dtype=mx.float32)
        shape_arr = mx.array([x_flat.size], dtype=mx.uint32)

        # Configure grid and threadgroup
        grid = (x_flat.size, 1, 1)
        threadgroup = (256, 1, 1)  # Standard threadgroup size

        # Call kernel with correct API
        kernel = self.compile()
        (out_flat,) = kernel(
            inputs=[x_flat, cap_arr, shape_arr],
            output_shapes=[x_flat.shape],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup,
        )

        # Reshape back to original shape
        return out_flat.reshape(x.shape)


# Functional interface for convenience
soft_cap = SoftCapMLXJitKernel()

__all__ = ['soft_cap']
