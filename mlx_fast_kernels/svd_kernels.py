"""
SVD subspace power-iteration kernels (baseline)

This module provides a simple, correct baseline kernel to compute:
    Z = Aᵀ (A V)
for a block of vectors V. It is intentionally naive (nested loops) and
serves as a contrast to the tiled two-kernel approach in `gemm_kernels.py`.

Notes
- The baseline uses a 1D launch over `n*k` and maps indices via `/` and `%` by
  a runtime `k`. This is pedagogical; the optimized path avoids such divides by
  using a 2D grid and shared-memory tiles.
- For real workloads, prefer the tiled approach (A@V then Aᵀ@B) exposed in
  `mlx_fast_kernels/gemm_kernels.py`.
"""

from __future__ import annotations

from typing import Tuple
from .gemm_kernels import gemm_av, gemm_at_b
import mlx.core as mx

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
