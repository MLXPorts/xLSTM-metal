"""
Baseline MLX/Metal kernel for Z = Aᵀ (A V).

Intentionally naive (1D grid, nested loops) and kept here for reference.
See production tiled path in `mlx_fast_kernels/svd_kernels.py:power_iter_step_tiled`.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_BODY_AT_A_V = r"""
    // Inputs: A (m,n), V (n,k), shape = [m, n, k]
    // Output: Z (n,k) = A^T (A V)
    uint gid = thread_position_in_grid.x;
    uint m = (uint)shape[0];
    uint n = (uint)shape[1];
    uint k = (uint)shape[2];
    uint total = n * k;
    if (gid >= total) return;

    uint col = gid % k;     // 0..k-1
    uint rowN = gid / k;    // 0..n-1

    float acc = 0.0f;
    for (uint i = 0; i < m; ++i) {
        float a_i_rowN = A[i * n + rowN];
        float av = 0.0f;
        for (uint j = 0; j < n; ++j) {
            av = fma(A[i * n + j], V[j * k + col], av);
        }
        acc = fma(a_i_rowN, av, acc);
    }
    Z[rowN * k + col] = acc;
"""

_KERNEL_AT_A_V = mx.fast.metal_kernel(name="svd_at_a_v", input_names=["A", "V", "shape"], output_names=["Z"],
                                      header=_HEADER, source=_BODY_AT_A_V)


def power_iter_step_naive(A: mx.array, V: mx.array) -> mx.array:
    """

    :param A:
    :param V:
    :return:
    """
    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(V.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)
    total = n * k
    tgroup = 256
    nthreads = ((total + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    threadgroup = (tgroup, 1, 1)
    (Z,) = _KERNEL_AT_A_V(
        inputs=[A, V, shape],
        output_shapes=[(n, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Z

