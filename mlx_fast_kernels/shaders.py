"""
MLX custom Metal kernels (body-only) for simple ops.

This module provides ready-to-use MLX kernels via `mx.fast.metal_kernel`
following the "body-only" authoring model:
- Soft-cap (elementwise clamp to +/- cap)
- Memcpy (elementwise copy)

Notes
- Only the kernel body is supplied; MLX auto-generates the full [[kernel]]
  signature and maps inputs/outputs based on input/output names.
- A tiny `shape` buffer (uint32) carries sizes for bounds checks.
- See docs/MLX_METAL_SHADER_INTEGRATION.md for details and patterns.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""


def _soft_cap_kernel():
    """Return a compiled body-only soft-cap kernel (cached per-process)."""
    src = r"""
    uint i = thread_position_in_grid.x;
    int size = int(shape[0]);
    if (i >= size) return;
    // Read input and scalar cap (as a single-element buffer)
    float x = inp[i];
    float c = cap[0];
    // Clamp to +/- c using tanh-based cap (or min/max form as needed)
    out[i] = c * tanh(x / c);
    """
    return mx.fast.metal_kernel(
        name="soft_cap",
        input_names=["inp", "cap", "shape"],
        output_names=["out"],
        header=_HEADER,
        source=src,
        ensure_row_contiguous=True,
    )


def _memcpy_kernel():
    """Return a compiled body-only memcpy kernel (cached per-process)."""
    src = r"""
    uint i = thread_position_in_grid.x;
    int size = int(shape[0]);
    if (i >= size) return;
    out[i] = inp[i];
    """
    return mx.fast.metal_kernel(
        name="memcpy_kernel",
        input_names=["inp", "shape"],
        output_names=["out"],
        header=_HEADER,
        source=src,
        ensure_row_contiguous=True,
    )


_K_SOFTCAP = None
_K_MEMCPY = None


def soft_cap(x: mx.array, cap_value: float) -> mx.array:
    """Apply an elementwise soft cap: out = cap * tanh(x/cap).

    Parameters
    - x: input array
    - cap_value: positive scalar cap (float)

    Returns
    - out: array with same shape/dtype as x
    """
    global _K_SOFTCAP
    if _K_SOFTCAP is None:
        _K_SOFTCAP = _soft_cap_kernel()
    n = int(x.size)
    shape = mx.array([n], dtype=mx.uint32)
    cap = mx.array([cap_value], dtype=x.dtype)
    (out,) = _K_SOFTCAP(
        inputs=[x, cap, shape],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n) or 1, 1, 1),
    )
    return out


def memcpy(x: mx.array) -> mx.array:
    """Elementwise copy via a body-only Metal kernel.

    Primarily for validation and examples; MLX assigns efficiently otherwise,
    but this demonstrates the `mx.fast.metal_kernel` contract.
    """
    global _K_MEMCPY
    if _K_MEMCPY is None:
        _K_MEMCPY = _memcpy_kernel()
    n = int(x.size)
    shape = mx.array([n], dtype=mx.uint32)
    (out,) = _K_MEMCPY(
        inputs=[x, shape],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n) or 1, 1, 1),
    )
    return out


__all__ = [
    "soft_cap",
    "memcpy",
]

