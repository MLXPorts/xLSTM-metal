
"""
MLX orthogonality helpers (left/right orthonormality + completion).

Implements a pure‑MLX two‑pass Modified Gram–Schmidt (MGS) QR factorization
to build orthonormal columns, utilities for orthonormal rows, and completion of
semi‑orthonormal bases to a full orthonormal set.

Notes
- Uses row‑major indexing and MX array ops; loops are over columns/rows only.
- Two‑pass MGS improves numerical stability over one‑pass and is often enough
  for basis construction without resorting to Householder.
- For larger panels or stricter numerics, pair with MLX Metal kernels in
  `mlx_fast_kernels.qr_kernels` for `c = Qᵀ v` and vector updates.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx


def _safe_norm(x: mx.array) -> mx.array:
    n = mx.sqrt(mx.sum(x * x))
    return mx.where(n > 0, n, mx.array(1.0, dtype=n.dtype))


def pure_mlx_qr(X: mx.array) -> Tuple[mx.array, mx.array]:
    """Compute a QR factorization (two-pass MGS) with MLX ops.

    Args
    - X: (m, n) input matrix

    Returns
    - Q: (m, n) with orthonormal columns (Q^T Q = I)
    - R: (n, n) upper-triangular
    """
    m, n = int(X.shape[0]), int(X.shape[1])
    Q = mx.zeros((m, n), dtype=X.dtype)
    R = mx.zeros((n, n), dtype=X.dtype)

    for j in range(n):
        v = X[:, j]
        # First pass
        for i in range(j):
            rij = mx.sum(Q[:, i] * v)
            R = R.at[i, j].set(rij)
            v -= rij * Q[:, i]
        # Second pass
        for i in range(j):
            rij2 = mx.sum(Q[:, i] * v)
            R = R.at[i, j].set(R[i, j] + rij2)
            v -= rij2 * Q[:, i]
        rjj = _safe_norm(v)
        R = R.at[j, j].set(rjj)
        qj = v / rjj
        Q = Q.at[:, j].set(qj)
    return Q, R


def orthonormal_columns(X: mx.array) -> mx.array:
    """Return Q with orthonormal columns spanning columns of X (left-orthonormal)."""
    Q, _ = pure_mlx_qr(X)
    return Q[:, : int(X.shape[1])]


def orthonormal_rows(X: mx.array) -> mx.array:
    """Return a matrix with orthonormal rows spanning rows of X (right-orthonormal)."""
    Qt, _ = pure_mlx_qr(mx.transpose(X))
    return mx.transpose(Qt[:, : int(X.shape[0])])


def complete_basis(Q: mx.array) -> mx.array:
    """Completes a semi-orthonormal basis to a full orthonormal basis.

    This function takes a matrix Q with orthonormal columns and appends new
    orthonormal columns to it to form a full orthonormal basis.

    Args:
        Q (mx.array): A matrix with orthonormal columns.

    Returns:
        A full orthonormal basis.
    """
    m, r = int(Q.shape[0]), int(Q.shape[1])
    k = m - r
    if k <= 0:
        return Q
    R = Q
    for _ in range(k):
        v = mx.random.normal(shape=(m,), dtype=R.dtype)
        # Two-pass MGS projection against existing columns in R
        c1 = mx.matmul(mx.transpose(R), v)
        v -= mx.matmul(R, c1)
        c2 = mx.matmul(mx.transpose(R), v)
        v -= mx.matmul(R, c2)
        nrm = _safe_norm(v)
        u = v / nrm
        R = mx.concatenate([R, u.reshape((m, 1))], axis=1)
    return R


__all__ = [
    "pure_mlx_qr",
    "orthonormal_columns",
    "orthonormal_rows",
    "complete_basis",
]
