<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

Comprehensive Guide to Using Metal in MLX (Correct, Working Patterns)

This guide consolidates the battle‑tested patterns we use with MLX’s `mx.fast.metal_kernel` on Apple GPUs. It aligns with MLX 0.29.x behavior and the conventions we verified in this repo. It complements Kernel-Guide.md and Metal-Primer.md with a full walkthrough and larger examples (Cholesky, QR, SVD).

1) MLX Metal Kernel API: What Actually Works

- Create once, reuse:
  ```python
  kernel = mx.fast.metal_kernel(
      name="my_kernel",
      input_names=["A", "shape"],
      output_names=["out"],
      header="#include <metal_stdlib>\nusing namespace metal;\n",
      source=r"""
          uint tid = thread_position_in_grid.x;
          uint m = (uint)shape[0];
          if (tid >= m) return;
          out[tid] = A[tid] + 1.0f;
      """,
      ensure_row_contiguous=True,
  )
  # Call (bind inputs/outputs, explicit launch sizes)
  (y,) = kernel(
      inputs=[x, mx.array([m], dtype=mx.uint32)],
      output_shapes=[(m,)],
      output_dtypes=[x.dtype],
      grid=(ceil_mul(m, 256), 1, 1),
      threadgroup=(256, 1, 1),
  )
  ```

- Header vs Source:
  - header: includes and helpers (branchless guards, reductions)
  - source: body‑only statements (no function signature)

- Input/Output names:
  - Names are exactly as provided in `input_names` / `output_names` (e.g., `A`, `shape`, `out`).
  - MLX also provides `<name>_shape` arrays (e.g., `A_shape[0]`) if needed, but the most robust pattern is to pass a dedicated small `shape` buffer explicitly.

- Recommended parameter passing:
  - Pack small buffers for shapes/flags/eps (e.g., `shape=[m,n,k]`, `flags=[use_eps_bits]`, `eps=[1e-6]`).
  - Avoid recompiling for shape changes by reusing the same kernel and feeding different buffers.

2) Thread Indices, Streams, and Launch Sizes

- Indices:
  - `thread_position_in_grid` (global), `threadgroup_position_in_grid` (block), `thread_position_in_threadgroup` (local)
  - Use 2D groups for tiles: `lid.x/lid.y`, `tg.x/tg.y`
- Sizing:
  - `threadgroup` ≤ 1024 threads; align x/y to 32 (Apple execution width)
  - Safe defaults for GEMM‑like: 16×16 (256 threads)
  - Use 1D for simple element‑wise ops

3) Synchronization and Reductions

- Barriers:
  - `threadgroup_barrier(mem_flags::mem_threadgroup)` for TG memory
  - `threadgroup_barrier(mem_flags::mem_device)` when reading/writing device buffers in phases
- Shared memory:
  - `threadgroup float tileA[TM][TN];` to stage tiles; barrier before FMA
- SIMD reductions:
  - Use `simd_sum(x)`, `simd_max(x)` to reduce within a warp; write one partial per warp to TG memory; combine on thread 0, then broadcast

4) Branchless Guards and “where” Equivalents

- Prefer ternary/clamp over divergent branches:
  - `x = (fabs(x) < eps) ? 0.0f : x;`
  - `x = fmin(fmax(x, -cap), cap);`
- Toggle behavior via a flags buffer (uint32 bitmask) rather than JITting new kernels.

5) Practical Examples

5.1 Element‑wise exp
```python
def exp_elementwise(a: mx.array) -> mx.array:
    header = """#include <metal_stdlib>\nusing namespace metal;\n"""
    source = r"""
        uint tid = thread_position_in_grid.x;
        uint n = (uint)shape[0];
        if (tid >= n) return;
        out[tid] = exp(a[tid]);
    """
    ker = mx.fast.metal_kernel(
        name="exp1d", input_names=["a", "shape"], output_names=["out"],
        header=header, source=source, ensure_row_contiguous=True)
    n = int(a.size)
    shape = mx.array([n], dtype=mx.uint32)
    (y,) = ker(inputs=[a, shape], output_shapes=[a.shape], output_dtypes=[a.dtype],
               grid=( (n+255)//256*256, 1, 1), threadgroup=(256,1,1))
    return y
```

5.2 Tiled GEMM core (A×B→C)
See `mlx_fast_kernels/gemm_kernels.py` for a working implementation:
- 16×16 tiles with TG memory staging
- Barriers between load/accumulate phases
- Coalesced loads and FMA across tile dimension

5.3 QR helpers
See `python/metalfaiss/faissmlx/kernels/qr_kernels.py`:
- `qr_col_dot`: c = Qᵀ v (column‑parallel dot)
- `qr_update_vec`: v ← v − Q c (row‑parallel update)
Both follow the header/body and explicit launch size patterns and pass `shape=[m,k]`.

6) Patterns that Don’t Help (or Aren’t Supported)

- JIT templating at call site: prefer runtime param buffers. Rebuilding kernels to toggle types or flags will thrash caches.
- Host scalar pulls inside hot paths: avoid `.item()`/`.numpy()`/`float()`/`int()` on MLX arrays; keep everything device‑resident.
- Global barriers: not available in MSL; structure multi‑phase algorithms as multiple kernels with natural synchronization points.

7) Block‑Based Algorithms

- Cholesky/QR: diagonal panel work is numerically sensitive (keep on fewer threads or single threadgroup); trailing updates are highly parallel (tile and fuse FMAs).
- SVD power iteration: we prefer two GEMM‑like kernels (A@V then Aᵀ@B) instead of a monolithic kernel — easier to tile, cache, and schedule.
- For advanced SVD strategies, including "banding" and multi-stream execution to improve cache locality and overlap work, see the project's [Research Journal](./../research/Journal.md).

8) Debugging and Diagnostics

- Start small: single thread or tiny tiles, then scale.
- Add optional `dbg` buffer (float) indexed with a few well‑known slots to capture early exit reasons and counts during bring‑up; remove or gate behind env in production.
- `mlx.core.metal.start_capture()` / `stop_capture()` can capture a small run for Xcode/GPU inspection.

9) Numerics

- Use branchless guards to avoid NaNs/inf; clamp tiny denominators.
- Where fp64 would help, consider compensated sums (e.g., a Kahan MLX helper) or limb techniques for critical inner products; keep them off the hot path unless needed.

10) Performance Considerations & Pitfalls

High-level performance tuning for Metal kernels involves a few key areas. For a deep dive, see [Shader-Optimization-Tips.md](./../metal/Shader-Optimization-Tips.md) and [WWDC16-Optimization-Patterns.md](./WWDC16-Optimization-Patterns.md).

- **Data Types:** Use the smallest practical data type (`half`, `short`) to improve register usage and ALU throughput.
- **Memory Access:** Avoid performance pitfalls like dynamically-indexed stack arrays, which can have a catastrophic impact. Ensure loads and stores are coalesced.
- **Integer Arithmetic:** Division or modulus by non-compile-time constants is extremely slow. Pre-calculate where possible.
- **Control Flow:** Prefer uniform control flow and use ternary operators over `if/else` where possible to avoid divergence.

11) Streams & Overlap (CPU/GPU)

- Use explicit streams to place independent work on separate queues (CPU vs GPU) and overlap compute with data prep; rely on MLX’s automatic cross‑stream dependency handling. Keep stream‑level sync to boundaries (e.g., logging, checkpoints).
- Pair compute streams with MLX Data streams (prefetch) to pipeline I/O/decoding and keep compute fed.
- See Streams-Guide: `docs/mlx_reference/Streams-Guide.md` for a plain‑speech walkthrough and examples.

12) Integration With Compiled MLX

- `mx.compile` can fuse MLX graphs (e.g., the MLX path of SVD Z‑step) and shrink Python overhead; shapes must be stable.
- Compiling won’t change the inner body of a custom Metal kernel, but a compiled wrapper can still reduce launch overhead when driving many kernels.

References

- This repo: Kernel-Guide.md, Metal-Primer.md, and working kernels in `python/metalfaiss/faissmlx/kernels/`.
- Spot tests: `docs/mlx_reference/Spot-Tests.md` — hands-on microbenchmarks to validate patterns in your environment.
- Ember ML backend (QR/Cholesky/SVD kernels) for examples of reductions, threading, and safety checks.
- Hardware-aware tiling: Kernels query `mlx.core.metal.device_info()` and accept env overrides (`METALFAISS_GEMM_TILE_*`). Benchmark tile shapes (16×16, 32×8, 8×32) per device.
