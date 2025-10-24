"""
Metal (MLX) kernels for Multi-Head LayerNorm (experimental, lab-only).

Implements a simdgroup-optimized per-head LayerNorm over DH:
- One warp (32 lanes) computes mean/var for a (B, NH) row across DH.
- Reductions use threadgroup memory and two barriers.

Contract
- Input X as (B, NH, DH) float32 (ensure row-contiguous).
- shape buffer: [B, NH, DH] (uint32)
- eps: single-element buffer (float32)
- Output Y as (B, NH, DH) float32 (normalized, no affine).

Notes
- Affine (gamma/beta) can be applied in Python if needed for parity testing.
- This is an experiment to assess kernel viability vs MLX ops.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_MHLN_SIMD_SRC = r"""
    const uint WARP = 32u;

    int B  = int(shape[0]);
    int NH = int(shape[1]);
    int DH = int(shape[2]);

    uint gid  = thread_position_in_grid.x; // global thread id
    uint lane = (gid & (WARP - 1u));      // 0..31
    int  row  = int(gid / WARP);          // one warp per (b, h)
    int  rows = B * NH;
    if (row >= rows) return;

    int b = row / NH;
    int h = row - b * NH;
    int base = (b * NH + h) * DH; // flatten index for (B,NH,DH)

    // First pass: mean
    float partial = 0.0f;
    for (int i = int(lane); i < DH; i += int(WARP)) {
        partial += X[base + i];
    }
    threadgroup float tmp[WARP];
    tmp[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float sum = 0.0f;
        for (uint l = 0u; l < WARP; ++l) sum += tmp[l];
        tmp[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = tmp[0] / float(DH);

    // Second pass: variance
    float vpart = 0.0f;
    for (int i = int(lane); i < DH; i += int(WARP)) {
        float d = X[base + i] - mean;
        vpart += d * d;
    }
    tmp[lane] = vpart;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float vsum = 0.0f;
        for (uint l = 0u; l < WARP; ++l) vsum += tmp[l];
        tmp[0] = vsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rstd = rsqrt(tmp[0] / float(DH) + eps[0]);

    // Write normalized output
    for (int i = int(lane); i < DH; i += int(WARP)) {
        float y = (X[base + i] - mean) * rstd;
        Y[base + i] = y;
    }
"""

_KERNEL_MHLN_SIMD = mx.fast.metal_kernel(
    name="mhln_simd",
    input_names=["X", "shape", "eps"],
    output_names=["Y"],
    source=_MHLN_SIMD_SRC,
    header=_HEADER,
    ensure_row_contiguous=True,
)


def mh_layernorm_simd(X: mx.array, eps: float = 1e-6) -> mx.array:
    """Compute per-head LayerNorm over DH with a simdgroup kernel.

    Parameters
    - X: (B, NH, DH) float32
    - eps: epsilon for numerical stability

    Returns
    - Y: (B, NH, DH) float32 normalized per head
    """
    assert X.ndim == 3, "X must be (B,NH,DH)"
    B, NH, DH = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    # Ensure dtype and shape buffers
    Xf = X.astype(mx.float32)
    shape = mx.array([B, NH, DH], dtype=mx.uint32)
    eps_arr = mx.array([eps], dtype=mx.float32)

    WARP = 32
    rows = B * NH
    grid = (rows * WARP, 1, 1)
    threadgroup = (WARP, 1, 1)

    (Y,) = _KERNEL_MHLN_SIMD(
        inputs=[Xf, shape, eps_arr],
        output_shapes=[(B, NH, DH)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Y


__all__ = [
    "mh_layernorm_simd",
]

