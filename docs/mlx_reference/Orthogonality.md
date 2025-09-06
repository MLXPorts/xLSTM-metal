<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

# Non‑Square Orthogonality (Left/Right + Completion)

Orthogonality isn’t just for square matrices. This note documents robust patterns for semi‑orthogonal matrices and completion to a full orthonormal basis, using MLX and GPU‑friendly kernels.

## Definitions

- Left‑orthonormal (orthonormal columns): Q ∈ R^{m×n}, m ≥ n, Q^T Q = I_n
- Right‑orthonormal (orthonormal rows): Q ∈ R^{m×n}, n ≥ m, Q Q^T = I_m

## Orthonormal Columns (Left)

```python
from metalfaiss.faissmlx.qr import pure_mlx_qr

def orthonormal_columns(X: mx.array) -> mx.array:
    # Two‑pass MGS via MLX QR builds Q with Q^T Q = I
    Q, _ = pure_mlx_qr(X)
    return Q[:, : X.shape[1]]
```

## Orthonormal Rows (Right)

```python
def orthonormal_rows(X: mx.array) -> mx.array:
    # Orthonormalize columns of X^T, then transpose back
    Qt, _ = pure_mlx_qr(mx.transpose(X))
    return mx.transpose(Qt[:, : X.shape[0]])
```

## Completing to a Full Basis

Append k = m − r new orthonormal columns to Q ∈ R^{m×r}:

```python
def complete_basis(Q: mx.array) -> mx.array:
    m, r = int(Q.shape[0]), int(Q.shape[1])
    k = m - r
    if k == 0:
        return Q
    R = Q
    for _ in range(k):
        v = mx.random.normal(shape=(m,), dtype=R.dtype)
        # two‑pass MGS projection
        c1 = mx.matmul(mx.transpose(R), v)
        v  = v - mx.matmul(R, c1)
        c2 = mx.matmul(mx.transpose(R), v)
        v  = v - mx.matmul(R, c2)
        nrm = mx.sqrt(mx.sum(v*v))
        u = v / mx.where(nrm > 0, nrm, 1)
        R = mx.concatenate([R, u.reshape((m, 1))], axis=1)
    return R
```

## GPU Notes

- Use the QR projection kernel (c = Q^T v) for large m,k to speed up re‑orthonormalization.
- Consider HPC16x8 limb accumulation for projections and norms when drift appears.
- Random rotations for non‑square transforms:
  - If d_in ≥ d_out: take first d_out columns of a left‑orthonormal Q.
  - If d_out > d_in: build right‑orthonormal rows and transpose.

