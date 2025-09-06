<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

Metal Primer for MLX + Metal Shading Language (MSL)

Overview

This primer distills the patterns we use to write fast, correct GPU kernels for MLX via `mx.fast.metal_kernel`. It focuses on Apple GPUs, threadgroups, simdgroup reductions, shared memory, and how to size kernels and pass parameters without recompiling.

Kernel Structure in MLX

- Header vs source:
  - header: `#include <metal_stdlib>` and `using namespace metal;`, plus any inline helpers (branchless guards, reductions).
  - source: body-only statements — no function signatures. MLX generates the kernel function and binds buffers.
- Buffers and shapes:
  - Use small MLX arrays for shapes/flags/eps to avoid recompiling per call.
  - Example: `shape = mx.array([m, n, k], dtype=mx.uint32)`; pass as an input buffer named `shape`.
- Launch configuration:
  - `grid=(gx, gy, gz)`, `threadgroup=(tx, ty, tz)`. Keep tx*ty*tz ≤ 1024; align tx, ty to 32 (execution width) where possible.
  - Use 2D threadgroups for tile work (e.g., 16×16) to improve locality.

Core Built-ins and Indices

- Thread indices:
  - `uint tid = thread_position_in_grid.x;`
  - `uint2 g = thread_position_in_grid.xy;`
  - `uint2 tg = threadgroup_position_in_grid.xy;`
  - `uint2 lid = thread_position_in_threadgroup.xy;`
- Sizes:
  - `threads_per_threadgroup`, `grid_size` — handy for reductions and circuit breakers.
- SIMD lanes:
  - `WARP_SIZE` is effectively 32 on Apple GPUs. Use simdgroup reductions.

Shared Memory and Reductions

- Declare threadgroup arrays for tiles and small scratch:
  - `threadgroup float Asub[TM][TN];`
  - `threadgroup float partial[32];`
- Synchronize within a threadgroup:
  - `threadgroup_barrier(mem_flags::mem_threadgroup);`
- SIMD reductions (warp-level):
  - Use `simd_sum(x)`, `simd_max(x)` to reduce across lanes; have lane 0 write a partial to threadgroup memory, then have thread 0 combine.

Branchless Guards and “where”

- Use ternary and clamp instead of divergent branches:
  - `float x_safe = (fabs(x) < eps) ? 0.0f : x;`
  - `x = fmin(fmax(x, -cap), cap);`
- For toggles, pass flags in a small uint32 buffer and create `bool use_eps = (flags & 1u) != 0;`.

Memory Model

- Global `device` buffers are bound by MLX (correspond to input_names/output_names order).
- Local arrays (registers) are per-thread; `threadgroup` arrays are shared in the workgroup.
- Avoid giant per-kernel resource counts: Apple Metal has a practical limit of 64 argument buffers — pack scalars into small arrays.

Sizing and Tiling

- GEMM-like kernels:
  - Tiles of 16×16 (256 threads) are a safe default; test 32×8 and 8×32 per device.
  - Stage tiles of A and B into threadgroup memory; barrier; FMA across tile dimension.
  - Coalesce loads: organize memory such that adjacent threads read adjacent elements.
- Reduction kernels (dot/norm):
  - Accumulate per-thread partials; reduce via `simd_sum`; write per-warp partials; combine on thread 0; broadcast results via threadgroup memory.

MLX Binding Cheatsheet

- Fast kernel creation:
  ```python
  header = """#include <metal_stdlib>\nusing namespace metal;\n"""
  src = r"""
      uint tid = thread_position_in_grid.x;
      uint m = shape[0]; uint n = shape[1];
      if (tid >= m) return;
      out[tid] = in0[tid] + in1[tid];
  """
  kernel = mx.fast.metal_kernel(
      name="add1d", input_names=["in0","in1","shape"], output_names=["out"],
      header=header, source=src, ensure_row_contiguous=True)
  (y,) = kernel(inputs=[x0,x1,shape], output_shapes=[(m,)], output_dtypes=[x0.dtype],
                grid=(ceil_mul(m, 64),1,1), threadgroup=(64,1,1))
  ```

Circuit Breakers and Diagnostics (Optional)

- Add a tiny `dbg` buffer to record flags and early exit reasons in debug builds:
  - `dbg[0] = 1.0f` at start; `dbg[13] = code` on failure; threadgroup-barrier before exit.
- Never leave heavy diag in hot paths.

Pitfalls

- No global barrier across the entire grid. If you need a two-phase algorithm (compute c then use c), either:
  - restrict to a single threadgroup (small problems), or
  - split into two kernel launches (what we generally do), or
  - stage per-block `c` into global memory and design the second phase to tolerate partially-filled tiles.
- Don’t rebuild kernels per call; pass shapes/flags in buffers instead.
- Keep kernel argument counts small — pack params.
- **Dynamically-indexed stack arrays** where the array itself is not a compile-time constant can have a "catastrophic" performance impact. Avoid them.
- **Integer division/modulus** by a denominator that is not a compile-time constant is extremely slow. Pre-calculate reciprocals or use bit shifts where possible.

Fusing Work (When Safe)

- For QR, projecting and updating in a single kernel requires either:
  - one threadgroup and a global barrier (not available), or
  - two kernels (what we do): dot then update; still a big win by avoiding Python overhead.
- For SVD Z-step, we prefer two GEMM-like kernels over a monolithic one; easier to tune and tile.

References

- Ember ML kernel code (QR/Cholesky/SVD) — rich examples of simd reductions and safety.
- Apple GPU execution width (32); MLX fast.metal_kernel API.
- See also: docs/mlx_reference/Kernel-Guide.md for end-to-end MLX + Metal usage.

