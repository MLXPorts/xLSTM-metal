<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

# MLX + Metal Kernel Guide (Practical)

This guide captures patterns that work reliably with `mlx.core.fast.metal_kernel`, plus GPU–tiling strategies and numerics that hold up on real Apple GPUs.

## fast.metal_kernel: Body Only + Header

- Put includes and namespaces into `header`; the `source` must be the kernel body (no function signature).
- Pass shapes via small buffers (`uint[N]`) instead of baking sizes into code.
- Configure grid/threadgroup explicitly; use execution–width multiples (32) and cap ≤ 1024 threads/tg.

Example (column–wise projection coefficients c = Q^T v):

```python
# python/metalfaiss/faissmlx/kernels/qr_kernels.py
header = """#include <metal_stdlib>\nusing namespace metal;\n"""
source = r"""
    uint gid = thread_position_in_grid.x;
    uint m = (uint)shape[0];
    uint k = (uint)shape[1];
    if (gid >= k) return;
    float acc = 0.0f;
    for (uint i = 0; i < m; ++i) {
        acc += Q[i * k + gid] * v[i];
    }
    out[gid] = acc;
""";

kernel = mx.fast.metal_kernel(
    name="qr_col_dot",
    input_names=["Q", "v", "shape"],
    output_names=["out"],
    header=header,
    source=source,
    ensure_row_contiguous=True,
)
```

Launch:

```python
m, k = int(Q.shape[0]), int(Q.shape[1])
shape = mx.array([m, k], dtype=mx.uint32)
total = k
tgroup = 64
nthreads = ((total + tgroup - 1) // tgroup) * tgroup
grid = (nthreads, 1, 1)
threadgroup = (tgroup, 1, 1)
(out,) = kernel(
    inputs=[Q, v, shape],
    output_shapes=[(k,)],
    output_dtypes=[Q.dtype],
    grid=grid,
    threadgroup=threadgroup,
)
```

## Autoswitch (Size/Device–Aware)

Select implementations based on device and problem size (mirrors robust patterns in `ember_ml`):

- Small/medium: MLX vectorized ops (no JIT latency; plenty fast).
- Large: tiled Metal kernels for inner loops (dot products, panel updates).
- Numerically tough tiles: limb–based accumulation (HPC16x8) for dot and norm.

Pseudo:

```python
def choose_qr_impl(m, k, dev):
    if m*k < 1<<18: return "MLX_MGS"
    if dev.is_gpu:   return "KERNEL_MGS"
    return "MLX_MGS"
```

## QR Orthonormalization (MGS, two passes)

- Use two–pass Modified Gram–Schmidt for stability at fp32.
- Offload `c = Q^T v` to the Metal kernel when it wins; update `v ← v − Qc` in MLX.

Snippet:

```python
# python/metalfaiss/faissmlx/qr.py (simplified)
Q = mx.zeros((m, m), dtype=A.dtype)
R = mx.zeros((m, n), dtype=A.dtype)
for k in range(min(m, n)):
    v = A[:, k]
    if k > 0:
        Qk = Q[:, :k]
        c1 = project_coeffs(Qk, v)  # kernel
        v  = v - mx.matmul(Qk, c1)
        c2 = project_coeffs(Qk, v)
        v  = v - mx.matmul(Qk, c2)
        R[:k, k] = c1 + c2
    rkk = mx.sqrt(mx.sum(v * v))
    qk  = v / mx.where(rkk > 0, rkk, 1)
    Q[:, k] = qk
    R[k, k] = rkk
```

## SVD (Top‑k, Subspace Power Iteration)

- Iterate Z = A^T(A V) and re‑orthonormalize V with QR.
- The baseline is MLX GEMM (`mx.matmul`), which is highly optimized.
- For more performance, the Z-step is implemented as two separate, tiled GEMM-like Metal kernels:
  1.  `B = A @ V`
  2.  `Z = A.T @ B`
- This two-kernel approach is easier to tile and optimize than a single monolithic kernel. For smaller `k`, a "banding" strategy that processes columns of `V` in smaller groups can further improve cache locality and performance.

Outline:

```python
V = orthonormal_columns(mx.random.normal((n, k)))
for _ in range(iters):
    # Z can be computed via MLX or a two-pass tiled Metal kernel
    AV = mx.matmul(A, V) 
    Z  = mx.matmul(A.T, AV)
    V, _ = pure_mlx_qr(Z)
U  = mx.matmul(A, V)
S  = mx.sqrt(mx.sum(U*U, axis=0))
U  = U / mx.where(S > 0, S, 1)[None, :]
```

## Performance Pitfalls

When writing Metal kernels, be aware of common performance pitfalls that can silently degrade performance:

- **Dynamically-Indexed Stack Arrays:** Avoid arrays on the stack that are indexed by a non-compile-time-constant value. This can prevent compiler optimizations and lead to significant slowdowns.
- **Non-Constant Integer Division:** Division or modulus operations where the denominator is not a compile-time constant are extremely slow on the GPU. Whenever possible, pre-calculate reciprocals and multiply, or use bit-shifting for powers of two.

For a more comprehensive list of optimizations, see [Shader-Optimization-Tips.md](../metal/Shader-Optimization-Tips.md) and [WWDC16-Optimization-Patterns.md](./WWDC16-Optimization-Patterns.md).

## Tile Selection (Hardware-Aware)

- Kernels in `gemm_kernels.py` select tile sizes at import using `mlx.core.metal.device_info()` and allow env overrides:
  - `METALFAISS_GEMM_TILE_AV="TMxT"` (AV kernel, TN=TK=T)
  - `METALFAISS_GEMM_TILE_ATB="TNxTK"` (AT_B kernel)
- Defaults: M3 → AV(32×8), AT_B(8×32); other Apple GPUs default to 16×16.
- Always benchmark on your device; (32,8) and (8,32) often compete with (16,16).

## HPC16x8 (128‑bit Limb Accumulation)

- When float32 accumulations drift (long dots, Gram updates), emulate extended precision via 16‑bit limbs:
  - Accumulate partial sums into 8×16‑bit limbs (radix 2^16) per thread/wave.
  - Reduce and carry–propagate to recover a high component; convert back to float32.
- Targeted use: projections `Q^T v`, vector norms, QR rank‑k updates.

## Non‑Square Orthogonality

- Left‑orthonormal (columns): Q ∈ R^{m×n}, Q^T Q = I_n.
- Right‑orthonormal (rows): Q ∈ R^{m×n}, Q Q^T = I_m.
- For completion: append random vectors, project out existing subspace with two‑pass MGS, normalize — repeat until full basis.

## Bench & Prune

- Always benchmark MLX vs kernel for your sizes.
- Keep one winner per path to simplify maintenance; re‑run benchmarks when shapes/devices change.

## Spot Tests (Learn by Measuring)

- For hands-on microbenches that illustrate key performance rules (integer division vs 2D grids, barrier scope, half I/O + float accumulate), see `docs/mlx_reference/Spot-Tests.md`.

## Streams (Overlap & Boundaries)

- Place independent tasks on explicit streams (CPU/GPU) to overlap work. Keep dependent steps in the same stream; synchronize only at program boundaries. See `docs/mlx_reference/Streams-Guide.md` for examples.
