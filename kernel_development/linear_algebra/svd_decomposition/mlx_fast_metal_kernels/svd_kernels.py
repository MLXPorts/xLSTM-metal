"""
SVD subspace power-iteration Z-step implementations (MLX + Metal).

Two variants are exposed:
- `power_iter_step_tiled(A, V)`: production path that computes Z = Aᵀ (A V)
  via two tiled GEMMs (A@V, then Aᵀ@B) using the kernels in
  `mlx_fast_kernels.gemm_kernels`. This avoids runtime divides/mods in hot
  loops, reuses data via threadgroup tiles, and honors MLX dispatch semantics.
- `power_iter_step(A, V)`: alias to the tiled path for clarity.

Notes
- A naive 1D kernel (Z = Aᵀ (A V)) is preserved in `research_archive` for
  teaching and reference; it uses nested loops and `/`/`%` to derive indices.
  The tiled path here should be used for real workloads.
"""

from __future__ import annotations

import mlx.core as mx

from kernel_development.matrix.gemm.mlx_fast_metal_kernel.gemm_kernels import gemm_av, gemm_at_b


def power_iter_step(A: mx.array, V: mx.array) -> mx.array:
    """Alias to the tiled Z-step for production: Z = Aᵀ (A V)."""
    return power_iter_step_tiled(A, V)


__all__ = [
    "power_iter_step",
    "power_iter_step_tiled",
]


def power_iter_step_tiled(A: mx.array, V: mx.array) -> mx.array:
    """Compute Z = Aᵀ (A V) using tiled GEMM kernels.

    Parameters
    - `A (m,n)`
    - `V (n,k)`

    Returns
    - `Z (n,k)`
    """
    # B = A @ V  (m,k)
    B = gemm_av(A, V)
    # Z = A^T @ B (n,k)
    Z = gemm_at_b(A, B)
    return Z
