# Comprehensive MLX + Metal Reference Guide

**Complete guide to using MLX with Metal on Apple GPUs**

This superdocument consolidates all MLX + Metal reference materials for this xLSTM project, covering kernel development,
optimization patterns, testing, and best practices. Each section preserves detailed examples and practical guidance.

---

## Table of Contents

1. [MLX Metal Kernel API & Core Patterns](#1-mlx-metal-kernel-api--core-patterns)
2. [Metal Primer for MLX](#2-metal-primer-for-mlx)
3. [Practical Kernel Guide](#3-practical-kernel-guide)
4. [WWDC16-Inspired Optimization Patterns](#4-wwdc16-inspired-optimization-patterns)
5. [Spot Tests: Teaching Microbenchmarks](#5-spot-tests-teaching-microbenchmarks)
6. [Streams: Practical Overlap and Concurrency](#6-streams-practical-overlap-and-concurrency)
7. [Non-Square Orthogonality](#7-non-square-orthogonality)
8. [Static Tuning Parameters](#8-static-tuning-parameters)
9. [External Reference Map](#9-external-reference-map)

---

## 1. MLX Metal Kernel API & Core Patterns

This section consolidates the battle-tested patterns we use with MLX's `mx.fast.metal_kernel` on Apple GPUs. It aligns
with MLX 0.29.x behavior and the conventions verified in this repo.

### 1.1 MLX Metal Kernel API: What Actually Works

**Create once, reuse:**

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

**Header vs Source:**

- header: includes and helpers (branchless guards, reductions)
- source: body-only statements (no function signature)

**Input/Output names:**

- Names are exactly as provided in `input_names` / `output_names` (e.g., `A`, `shape`, `out`).
- MLX also provides `<name>_shape` arrays (e.g., `A_shape[0]`) if needed, but the most robust pattern is to pass a
  dedicated small `shape` buffer explicitly.

**Recommended parameter passing:**

- Pack small buffers for shapes/flags/eps (e.g., `shape=[m,n,k]`, `flags=[use_eps_bits]`, `eps=[1e-6]`).
- Avoid recompiling for shape changes by reusing the same kernel and feeding different buffers.

### 1.2 Thread Indices, Streams, and Launch Sizes

**Indices:**

- `thread_position_in_grid` (global), `threadgroup_position_in_grid` (block), `thread_position_in_threadgroup` (local)
- Use 2D groups for tiles: `lid.x/lid.y`, `tg.x/tg.y`

**Sizing:**

- `threadgroup` ≤ 1024 threads; align x/y to 32 (Apple execution width)
- Safe defaults for GEMM-like: 16×16 (256 threads)
- Use 1D for simple element-wise ops

### 1.3 Synchronization and Reductions

**Barriers:**

- `threadgroup_barrier(mem_flags::mem_threadgroup)` for TG memory
- `threadgroup_barrier(mem_flags::mem_device)` when reading/writing device buffers in phases

**Shared memory:**

- `threadgroup float tileA[TM][TN];` to stage tiles; barrier before FMA

**SIMD reductions:**

- Use `simd_sum(x)`, `simd_max(x)` to reduce within a warp; write one partial per warp to TG memory; combine on thread
  0, then broadcast

### 1.4 Branchless Guards and "where" Equivalents

Prefer ternary/clamp over divergent branches:

- `x = (fabs(x) < eps) ? 0.0f : x;`
- `x = fmin(fmax(x, -cap), cap);`

Toggle behavior via a flags buffer (uint32 bitmask) rather than JITting new kernels.

### 1.5 Practical Examples

**5.1 Element-wise exp**

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

**5.2 Tiled GEMM core (A×B→C)**
See `mlx_fast_kernels/gemm_kernels.py` for a working implementation:

- 16×16 tiles with TG memory staging
- Barriers between load/accumulate phases
- Coalesced loads and FMA across tile dimension

**5.3 QR helpers**
See `python/metalfaiss/faissmlx/kernels/qr_kernels.py`:

- `qr_col_dot`: c = Qᵀ v (column-parallel dot)
- `qr_update_vec`: v ← v − Q c (row-parallel update)
  Both follow the header/body and explicit launch size patterns and pass `shape=[m,k]`.

### 1.6 Patterns that Don't Help (or Aren't Supported)

- **JIT templating at call site:** prefer runtime param buffers. Rebuilding kernels to toggle types or flags will thrash
  caches.
- **Host scalar pulls inside hot paths:** avoid `.item()`/`.numpy()`/`float()`/`int()` on MLX arrays; keep everything
  device-resident.
- **Global barriers:** not available in MSL; structure multi-phase algorithms as multiple kernels with natural
  synchronization points.

### 1.7 Block-Based Algorithms

- **Cholesky/QR:** diagonal panel work is numerically sensitive (keep on fewer threads or single threadgroup); trailing
  updates are highly parallel (tile and fuse FMAs).
- **SVD power iteration:** we prefer two GEMM-like kernels (A@V then Aᵀ@B) instead of a monolithic kernel — easier to
  tile, cache, and schedule.
- For advanced SVD strategies, including "banding" and multi-stream execution to improve cache locality and overlap
  work, see the project's Research Journal.

### 1.8 Debugging and Diagnostics

- Start small: single thread or tiny tiles, then scale.
- Add optional `dbg` buffer (float) indexed with a few well-known slots to capture early exit reasons and counts during
  bring-up; remove or gate behind env in production.
- `mlx.core.metal.start_capture()` / `stop_capture()` can capture a small run for Xcode/GPU inspection.

### 1.9 Numerics

- Use branchless guards to avoid NaNs/inf; clamp tiny denominators.
- Where fp64 would help, consider compensated sums (e.g., a Kahan MLX helper) or limb techniques for critical inner
  products; keep them off the hot path unless needed.

### 1.10 Performance Considerations & Pitfalls

High-level performance tuning for Metal kernels involves a few key areas. For a deep dive, see sections 4 and 5 of this
guide.

- **Data Types:** Use the smallest practical data type (`half`, `short`) to improve register usage and ALU throughput.
- **Memory Access:** Avoid performance pitfalls like dynamically-indexed stack arrays, which can have a catastrophic
  impact. Ensure loads and stores are coalesced.
- **Integer Arithmetic:** Division or modulus by non-compile-time constants is extremely slow. Pre-calculate where
  possible.
- **Control Flow:** Prefer uniform control flow and use ternary operators over `if/else` where possible to avoid
  divergence.

### 1.11 Streams & Overlap (CPU/GPU)

- Use explicit streams to place independent work on separate queues (CPU vs GPU) and overlap compute with data prep;
  rely on MLX's automatic cross-stream dependency handling. Keep stream-level sync to boundaries (e.g., logging,
  checkpoints).
- Pair compute streams with MLX Data streams (prefetch) to pipeline I/O/decoding and keep compute fed.
- See Section 6 for a plain-speech walkthrough and examples.

### 1.12 Integration With Compiled MLX

- `mx.compile` can fuse MLX graphs (e.g., the MLX path of SVD Z-step) and shrink Python overhead; shapes must be stable.
- Compiling won't change the inner body of a custom Metal kernel, but a compiled wrapper can still reduce launch
  overhead when driving many kernels.

---

## 2. Metal Primer for MLX

This primer distills the patterns we use to write fast, correct GPU kernels for MLX via `mx.fast.metal_kernel`. It
focuses on Apple GPUs, threadgroups, simdgroup reductions, shared memory, and how to size kernels and pass parameters
without recompiling.

### 2.1 Kernel Structure in MLX

**Header vs source:**

- header: `#include <metal_stdlib>` and `using namespace metal;`, plus any inline helpers (branchless guards,
  reductions).
- source: body-only statements — no function signatures. MLX generates the kernel function and binds buffers.

**Buffers and shapes:**

- Use small MLX arrays for shapes/flags/eps to avoid recompiling per call.
- Example: `shape = mx.array([m, n, k], dtype=mx.uint32)`; pass as an input buffer named `shape`.

**Launch configuration:**

- `grid=(gx, gy, gz)`, `threadgroup=(tx, ty, tz)`. Keep tx*ty*tz ≤ 1024; align tx, ty to 32 (execution width) where
  possible.
- Use 2D threadgroups for tile work (e.g., 16×16) to improve locality.

### 2.2 Core Built-ins and Indices

**Thread indices:**

- `uint tid = thread_position_in_grid.x;`
- `uint2 g = thread_position_in_grid.xy;`
- `uint2 tg = threadgroup_position_in_grid.xy;`
- `uint2 lid = thread_position_in_threadgroup.xy;`

**Sizes:**

- `threads_per_threadgroup`, `grid_size` — handy for reductions and circuit breakers.

**SIMD lanes:**

- `WARP_SIZE` is effectively 32 on Apple GPUs. Use simdgroup reductions.

### 2.3 Shared Memory and Reductions

**Declare threadgroup arrays for tiles and small scratch:**

- `threadgroup float Asub[TM][TN];`
- `threadgroup float partial[32];`

**Synchronize within a threadgroup:**

- `threadgroup_barrier(mem_flags::mem_threadgroup);`

**SIMD reductions (warp-level):**

- Use `simd_sum(x)`, `simd_max(x)` to reduce across lanes; have lane 0 write a partial to threadgroup memory, then have
  thread 0 combine.

### 2.4 Branchless Guards and "where"

Use ternary and clamp instead of divergent branches:

- `float x_safe = (fabs(x) < eps) ? 0.0f : x;`
- `x = fmin(fmax(x, -cap), cap);`

For toggles, pass flags in a small uint32 buffer and create `bool use_eps = (flags & 1u) != 0;`.

### 2.5 Memory Model

- Global `device` buffers are bound by MLX (correspond to input_names/output_names order).
- Local arrays (registers) are per-thread; `threadgroup` arrays are shared in the workgroup.
- Avoid giant per-kernel resource counts: Apple Metal has a practical limit of 64 argument buffers — pack scalars into
  small arrays.

### 2.6 Sizing and Tiling

**GEMM-like kernels:**

- Tiles of 16×16 (256 threads) are a safe default; test 32×8 and 8×32 per device.
- Stage tiles of A and B into threadgroup memory; barrier; FMA across tile dimension.
- Coalesce loads: organize memory such that adjacent threads read adjacent elements.

**Reduction kernels (dot/norm):**

- Accumulate per-thread partials; reduce via `simd_sum`; write per-warp partials; combine on thread 0; broadcast results
  via threadgroup memory.

### 2.7 MLX Binding Cheatsheet

Fast kernel creation:

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

### 2.8 Circuit Breakers and Diagnostics (Optional)

Add a tiny `dbg` buffer to record flags and early exit reasons in debug builds:

- `dbg[0] = 1.0f` at start; `dbg[13] = code` on failure; threadgroup-barrier before exit.
- Never leave heavy diag in hot paths.

### 2.9 Pitfalls

- **No global barrier** across the entire grid. If you need a two-phase algorithm (compute c then use c), either:
    - restrict to a single threadgroup (small problems), or
    - split into two kernel launches (what we generally do), or
    - stage per-block `c` into global memory and design the second phase to tolerate partially-filled tiles.
- Don't rebuild kernels per call; pass shapes/flags in buffers instead.
- Keep kernel argument counts small — pack params.
- **Dynamically-indexed stack arrays** where the array itself is not a compile-time constant can have a "catastrophic"
  performance impact. Avoid them.
- **Integer division/modulus** by a denominator that is not a compile-time constant is extremely slow. Pre-calculate
  reciprocals or use bit shifts where possible.

### 2.10 Fusing Work (When Safe)

For QR, projecting and updating in a single kernel requires either:

- one threadgroup and a global barrier (not available), or
- two kernels (what we do): dot then update; still a big win by avoiding Python overhead.

For SVD Z-step, we prefer two GEMM-like kernels over a monolithic one; easier to tune and tile.

---

## 3. Practical Kernel Guide

This guide captures patterns that work reliably with `mlx.core.fast.metal_kernel`, plus GPU-tiling strategies and
numerics that hold up on real Apple GPUs.

### 3.1 fast.metal_kernel: Body Only + Header

- Put includes and namespaces into `header`; the `source` must be the kernel body (no function signature).
- Pass shapes via small buffers (`uint[N]`) instead of baking sizes into code.
- Configure grid/threadgroup explicitly; use execution-width multiples (32) and cap ≤ 1024 threads/tg.

**Example (column-wise projection coefficients c = Q^T v):**

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

**Launch:**

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

### 3.2 Autoswitch (Size/Device-Aware)

Select implementations based on device and problem size (mirrors robust patterns in `ember_ml`):

- Small/medium: MLX vectorized ops (no JIT latency; plenty fast).
- Large: tiled Metal kernels for inner loops (dot products, panel updates).
- Numerically tough tiles: limb-based accumulation (HPC16x8) for dot and norm.

Pseudo:

```python
def choose_qr_impl(m, k, dev):
    if m*k < 1<<18: return "MLX_MGS"
    if dev.is_gpu:   return "KERNEL_MGS"
    return "MLX_MGS"
```

### 3.3 QR Orthonormalization (MGS, two passes)

- Use two-pass Modified Gram-Schmidt for stability at fp32.
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

### 3.4 SVD (Top-k, Subspace Power Iteration)

- Iterate Z = A^T(A V) and re-orthonormalize V with QR.
- The baseline is MLX GEMM (`mx.matmul`), which is highly optimized.
- For more performance, the Z-step is implemented as two separate, tiled GEMM-like Metal kernels:
    1. `B = A @ V`
    2. `Z = A.T @ B`
- This two-kernel approach is easier to tile and optimize than a single monolithic kernel. For smaller `k`, a "banding"
  strategy that processes columns of `V` in smaller groups can further improve cache locality and performance.

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

### 3.5 Performance Pitfalls

When writing Metal kernels, be aware of common performance pitfalls that can silently degrade performance:

- **Dynamically-Indexed Stack Arrays:** Avoid arrays on the stack that are indexed by a non-compile-time-constant value.
  This can prevent compiler optimizations and lead to significant slowdowns.
- **Non-Constant Integer Division:** Division or modulus operations where the denominator is not a compile-time constant
  are extremely slow on the GPU. Whenever possible, pre-calculate reciprocals and multiply, or use bit-shifting for
  powers of two.

For a more comprehensive list of optimizations, see Section 4 of this guide.

### 3.6 Tile Selection (Hardware-Aware)

- Kernels in `gemm_kernels.py` select tile sizes at import using `mlx.core.metal.device_info()` and allow env overrides:
    - `METALFAISS_GEMM_TILE_AV="TMxT"` (AV kernel, TN=TK=T)
    - `METALFAISS_GEMM_TILE_ATB="TNxTK"` (AT_B kernel)
- Defaults: M3 → AV(32×8), AT_B(8×32); other Apple GPUs default to 16×16.
- Always benchmark on your device; (32,8) and (8,32) often compete with (16,16).

### 3.7 HPC16x8 (128-bit Limb Accumulation)

- When float32 accumulations drift (long dots, Gram updates), emulate extended precision via 16-bit limbs:
    - Accumulate partial sums into 8×16-bit limbs (radix 2^16) per thread/wave.
    - Reduce and carry-propagate to recover a high component; convert back to float32.
- Targeted use: projections `Q^T v`, vector norms, QR rank-k updates.

### 3.8 Non-Square Orthogonality

- Left-orthonormal (columns): Q ∈ R^{m×n}, Q^T Q = I_n.
- Right-orthonormal (rows): Q ∈ R^{m×n}, Q Q^T = I_m.
- For completion: append random vectors, project out existing subspace with two-pass MGS, normalize — repeat until full
  basis.

### 3.9 Bench & Prune

- Always benchmark MLX vs kernel for your sizes.
- Keep one winner per path to simplify maintenance; re-run benchmarks when shapes/devices change.

### 3.10 Spot Tests (Learn by Measuring)

For hands-on microbenches that illustrate key performance rules (integer division vs 2D grids, barrier scope, half I/O +
float accumulate), see Section 5 of this guide.

### 3.11 Streams (Overlap & Boundaries)

Place independent tasks on explicit streams (CPU/GPU) to overlap work. Keep dependent steps in the same stream;
synchronize only at program boundaries. See Section 6 for examples.

---

## 4. WWDC16-Inspired Optimization Patterns

This guide translates concrete shader optimization patterns from Apple's "Advanced Metal Shader Optimization" (WWDC16
Session 606) into practices applicable to MLX's `mx.fast.metal_kernel`. It focuses on patterns for our math kernels (
QR/SVD/GEMM) and highlights how MLX's abstractions map to low-level Metal concepts.

### 4.1 Address Spaces And "Constant Buffer" Ideas

**Device vs constant:**

- Metal offers `device` (read/write) and `constant` (read-only, cached) address spaces. Small, read-only data (e.g.,
  shapes/flags) belongs in `constant` for preloading and reuse.
- MLX binds your inputs as buffers under the hood; you don't control qualifiers from Python. Practical adaptation:
    - Pack tiny params (shape, flags, eps) in small arrays and load them once into registers at the top of the kernel
      body.
    - Example (inside kernel source): `int m = int(shape[0]), n = int(shape[1]), k = int(shape[2]);`
    - Keep access statically bounded and avoid pointer chasing.

### 4.2 Compute Kernel Organization (Amortize Launch Overhead)

**Do enough work per thread:**

- Process 2 outputs per thread (e.g., two columns in GEMM) if register pressure allows.
- Reuse loaded tiles: accumulate into multiple accumulators `acc0, acc1`.
- Trade-off: more registers lowers occupancy; measure before committing.

**Split phases across kernels instead of global barriers:**

- No device-wide barrier in MSL. Use two kernels (e.g., dot then update for QR; A@V then Aᵀ@B for SVD) for clear sync
  points and simpler tuning.

### 4.3 Barriers: Use The Smallest Scope

- Prefer `threadgroup_barrier(mem_flags::mem_threadgroup)` when sharing TG arrays; use `mem_device` only when
  reading/writing global buffers across phases.
- If you can constrain to a single warp/group, `simdgroup_barrier` can be cheaper. In practice we tile with 16×16 TGs
  and use TG barriers.

### 4.4 Data Types (Register Footprint, ALU Throughput)

- Apple GPUs use 16-bit register units; smaller types can improve occupancy:
    - Consider `half` for intermediate math that tolerates reduced precision; keep accumulators in `float`.
    - Use `ushort` for local IDs where appropriate; we generally keep indices as `int` for safe addressing.
    - Avoid mixing `half` with float literals (`2.0`); use `half` literals (`2.0h`).

### 4.5 Arithmetic (Built-ins, Fast-Math)

- Fast-math is on by default; take advantage of:
    - `fma(a,b,c)` (fused multiply-add) — we adopted this in GEMM tiles and QR update.
    - Built-ins like `abs`, `saturate` are free modifiers; prefer them to manual code.
- Integer division/modulus:
    - Avoid divides by non-constants in hot loops. Precompute reciprocals and multiply, replace /2ⁿ with shifts. MLX
      doesn't expose function constants; prefer arithmetic transforms.

### 4.6 Control Flow (Uniform vs Divergent)

- Prefer uniform control flow across a warp; divergent branches serialize.
- Use ternary (select) for fast branchless decisions:
    - `x = cond ? a : b;`
    - Avoid "multiply by 0/1" tricks.

### 4.7 Memory Access (Vectorization, Stack, Addressing)

- Coalesce loads/stores; stage tiles in TG memory.
- Arrange structs for vectorizable access (SoA > AoS in many kernels).
- **Avoid dynamically-indexed, non-constant stack arrays.** The performance cost can be **catastrophic**. The WWDC
  session noted a real-world app that lost **30% of its performance** due to a single 32-byte dynamically indexed array.
  Compilers may unroll fixed-size loops to eliminate this, but it's a major pitfall to avoid.
- Use `int` (or smaller) for device memory addressing; prefer signed `int` over `uint` in index math to avoid extra
  instructions.

### 4.8 Latency, Occupancy, And Hiding

- Threadgroup memory and registers cap occupancy. Keep TG arrays modest and avoid excessive accumulators.
- Interleave independent work to hide latency if texture/long ops appear; for our math kernels, tiling + FMA dominates.

### 4.9 Putting It Into Practice (Our Kernels)

**GEMM tiles (A@V; Aᵀ@B):**

- 16×16 tiles; TG arrays for A and V/B; barriers between load/accumulate; explicit `fma` in inner loop; `int` indices
  for addressing.
- Try 32×8 or 8×32 on your device.

**QR helpers:**

- Separate kernels for dot and update; both use `fma`; leave projection norms to MLX unless reductions are a bottleneck.

**SVD Z-step:**

- Two GEMM-like kernels beat a monolith for maintainability and tuning. Banded processing can reduce peak memory;
  streams only help at large sizes; benchmark first.

### 4.10 Checklists

Before optimizing, ask:

- Are loads coalesced? Are you staging into TG memory?
- Is there an obvious `fma` opportunity?
- Is integer division avoidable? Can you precompute factors?
- Are your barriers the smallest scope that works?
- Are per-thread workloads large enough to amortize launch overhead without crushing occupancy?
- Are you using the smallest practical data types (`half`, `short`)?
- Have you eliminated dynamically-indexed stack arrays?

---

## 5. Spot Tests: Teaching Microbenchmarks

### 5.1 Purpose

These small, focused tests demonstrate why certain optimization patterns matter. Each test includes: what to change, how
to measure, and what to look for. Run them on your machine and compare wall time; optionally capture one representative
run in Xcode to inspect counters.

### 5.2 How to Run

- Use MLX's `mx.fast.metal_kernel` as shown. Each snippet compiles a kernel once and launches it repeatedly.
- Measure with `time.perf_counter()` and force evaluation via `mx.eval(...)`.
- Prefer a size that is large enough to be compute-bound but quick to iterate (e.g., 1–10M elements on your GPU).

### 5.3 Test 1: Avoid Integer Division/Modulus by Non-Constants

**Idea:**
Re-map 1D thread IDs to a 2D grid using `/` and `%` is slow when the divisor isn't a compile-time constant. Prefer 2D
grids or precomputed strides.

**Kernels:**

```python
import mlx.core as mx, time

header = """#include <metal_stdlib>\nusing namespace metal;\n"""

# Slow: 1D with / and % by runtime k
src_div = r"""
  uint gid = thread_position_in_grid.x;
  uint n = shape[0];
  uint k = shape[1];
  uint total = n * k;
  if (gid >= total) return;
  uint col = gid % k;     // non-constant modulus
  uint row = gid / k;     // non-constant division
  out[gid] = in0[row * k + col] * 2.0f;
"""

# Fast: 2D grid, no division/modulus
src_2d = r"""
  uint2 g = thread_position_in_grid.xy;
  uint n = shape[0];
  uint k = shape[1];
  if (g.x >= k || g.y >= n) return;
  uint idx = g.y * k + g.x;
  out[idx] = in0[idx] * 2.0f;
"""

def bench_div_vs_2d(n=2048, k=2048, reps=5):
  arr = mx.random.normal((n*k,)).astype(mx.float32)
  shape = mx.array([n,k], dtype=mx.uint32)
  ker_div = mx.fast.metal_kernel(name="divmap", input_names=["in0","shape"], output_names=["out"], header=header, source=src_div, ensure_row_contiguous=True)
  ker_2d  = mx.fast.metal_kernel(name="twod",   input_names=["in0","shape"], output_names=["out"], header=header, source=src_2d,  ensure_row_contiguous=True)
  # Warm
  (y,) = ker_div(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=((n*k+255)//256*256,1,1), threadgroup=(256,1,1)); mx.eval(y)
  (y,) = ker_2d(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=( (k+31)//32*32, (n+31)//32*32,1), threadgroup=(32,32,1)); mx.eval(y)
  def timeit(f):
    ts=[]
    for _ in range(reps):
      t0=time.perf_counter(); (y,)=f(); mx.eval(y); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]
  t_div = timeit(lambda: ker_div(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=((n*k+255)//256*256,1,1), threadgroup=(256,1,1)))
  t_2d  = timeit(lambda: ker_2d(inputs=[arr, shape], output_shapes=[arr.shape], output_dtypes=[arr.dtype], grid=( (k+31)//32*32, (n+31)//32*32,1), threadgroup=(32,32,1)))
  print(f"int-div map: {t_div:.4f}s; 2D map: {t_2d:.4f}s")
```

**What to expect:**
The 2D version should be faster when k is not a compile-time literal. If you must stick to 1D, precompute strides or
choose k as a function constant.

### 5.4 Test 2: Dynamic Stack Arrays vs Compile-Time Constant Arrays

**Idea:**
Dynamically-indexed, non-constant stack arrays can be catastrophic. If you can, restructure to constant-sized arrays or
unrolled loops.

**Kernels (illustrative):**

```cpp
// Bad: dynamic index into non-constant array (don’t do this in hot loops)
int foo(int a, int b, int c) {
  int tmp[2] = { a, b };
  return tmp[c];
}
// Okay: fixed-size loop; compiler can unroll
int sum3(int a, int b, int c) {
  int tmp3[3] = { a, b, c };
  int s = 0; for (int i=0;i<3;++i) s += tmp3[i]; return s;
}
```

**How to measure:**
If you have a real kernel that uses dynamic stack arrays, replace the pattern with a constant-sized array or a small
unrolled loop and re-measure. Expect substantial speedups if this was on a hot path.

### 5.5 Test 3: Barriers - threadgroup_barrier vs simdgroup_barrier

**Idea:**
Use the smallest scope that is correct. For warp-only reductions that never touch threadgroup memory across warps,
`simdgroup_barrier` can be cheaper.

**Sketch:**

```cpp
// Warp-only reduction
float x = ...;
float r = simd_sum(x);
// Synchronize lanes within the warp (no TG memory used across warps)
simdgroup_barrier(mem_flags::mem_none);
```

**Caution:**
If you stage data in `threadgroup` memory across multiple warps (e.g., 16×16 tiles), you must use
`threadgroup_barrier(mem_flags::mem_threadgroup)`.

### 5.6 Test 4: half I/O with float Accumulation

**Idea:**
Reduce bandwidth with `half` for loads/stores, but keep accumulators in `float`.

**Kernel snippet:**

```cpp
half ha = in_h[i];
float a = float(ha);
acc = fma(a, b, acc);
out_h[i] = half(acc);
```

**How to measure:**
Compare float32 end-to-end vs half I/O + float accumulate in a bandwidth-sensitive kernel (elementwise ops or GEMM with
small arithmetic intensity). Validate error bounds are acceptable.

### 5.7 Test 5: Branchless Select vs if/else

**Idea:**
Prefer fast ternary/select in simple guards.

**Kernel snippet:**

```cpp
// Branchless
float y = cond ? a : b;
```

**How to measure:**
On a kernel with simple clamping/guards, compare if/else vs ternary. The difference may be small in isolation; the main
benefit is avoiding divergence and enabling better instruction scheduling.

### 5.8 Appendix: Xcode GPU Capture

For any test, capture one run and check: shader time, memory transactions, and occupancy. Confirm qualitative
expectations (fewer scalar loads, no unexpected barriers).

---

## 6. Streams: Practical Overlap and Concurrency

### 6.1 Why Streams Matter

Streams let you run independent work in parallel (CPU and GPU) instead of queueing everything on the default stream.
That overlap is how you cut idle time and lift throughput.

### 6.2 Core Habits

- Put independent tasks on their own streams. If you skip the `stream=` argument, ops run on the default stream and can
  serialize.
- Keep a small, consistent set of streams per device (e.g., one or two for CPU work, one or two for GPU-heavy kernels).
  Predictability beats churning streams.
- Synchronize only at clear boundaries (e.g., logging, checkpoints, timed sections). Don't fence after every op.

### 6.3 Essentials

- Every op (including RNG) accepts a `stream` (or a `device`). Passing a `device` runs on that device's default stream;
  pass a `stream` for fine-grained control.
- MLX tracks cross-stream dependencies for you: if a result from stream A is used on stream B, MLX inserts the minimal
  wait.
- Use context managers to set scoped defaults so code stays readable.

### 6.4 Simple Example: CPU–GPU Overlap with Automatic Handoff

```python
import mlx.core as mx

# Create one CPU stream and one GPU stream
s_cpu = mx.new_stream(mx.cpu)
s_gpu = mx.new_stream(mx.gpu)

with mx.stream(s_cpu):
    x = mx.add(a, b)          # runs on CPU stream

with mx.stream(s_gpu):
    y = mx.multiply(x, b)     # runs on GPU stream; MLX waits only if x isn't ready

# Synchronize at a boundary (e.g., before reading y for logging)
mx.synchronize(s_gpu)
```

### 6.5 Callbacks Without Blocking the UI or Other Streams

Wait on a specific stream in the background, then fire a host callback. This scopes the wait and keeps other streams
running.

```python
from concurrent.futures import ThreadPoolExecutor
from docs.components.kernels.mlx_streams import on_stream_complete

s_gpu = mx.new_stream(mx.gpu)


def on_done():
    print("gpu stream finished a step")


# Wait in a worker, then call the callback
on_stream_complete(s_gpu, on_done, executor=ThreadPoolExecutor(max_workers=1))
```

### 6.6 Asyncio Integration

If you use an async app loop, do the wait in `run_in_executor` to avoid blocking the event loop.

```python
import asyncio
from docs.components.kernels.mlx_streams import on_stream_complete_async

await on_stream_complete_async(s_gpu, lambda: print("step done"))
```

### 6.7 Trigger by Array Evaluation

When the trigger is "these results are ready", evaluate them in a worker thread, then call the callback.

```python
from docs.components.kernels.mlx_streams import after_eval

future = after_eval([y], lambda: print("y ready"))
```

### 6.8 Pipelines and Prefetch

- Combine compute streams with MLX Data streams to keep the GPU fed while the CPU decodes and augments batches.
- Treat the data path as a stream early, then prefetch.

**Sketch:**

```python
# Pseudocode: use mlx-data to stream batches; overlap with GPU compute stream
ds = buffer.to_stream()         # turn buffer into a data stream
ds = ds.batch(32).prefetch(8, 4)

s_gpu = mx.new_stream(mx.gpu)
for batch in ds:
    with mx.stream(s_gpu):
        logits = model_forward(batch)
        loss = loss_fn(logits, batch.labels)
    # synchronize here only if you need to log/step synchronously
```

### 6.9 Synchronization Strategy

- Prefer stream-scoped `mx.synchronize(s)` at boundaries. Avoid global `mx.synchronize()` unless you truly want to stall
  the default device's default stream.
- Keep dependent steps in the same stream to benefit from in-stream ordering. Split only truly independent work across
  streams.

### 6.10 Determinism and RNG

- RNG ops honor the `stream` argument. Keep RNG on a stable stream mapping for reproducibility.
- If RNG values cross streams, MLX still orders the dependency; just avoid reshuffling streams mid-run if you care about
  exact repeatability.

### 6.11 Note

This repository does not include an MLX Swift implementation. All examples and APIs here refer to MLX Python.

### 6.12 Checklist

- Define a small, fixed set of streams per device.
- Use data streams with prefetch to keep compute streams busy.
- Synchronize only where results cross program boundaries.
- Rely on MLX's cross-stream dependency tracking rather than adding global fences.

### 6.13 Where to Find the Helpers

`tools/mlx_streams.py` provides `on_stream_complete`, `on_stream_complete_async`, and `after_eval` so you don't have to
re-write the patterns.

### 6.14 Common Pitfalls

- Letting everything fall onto the default stream (lost overlap, hidden contention).
- Sprinkling `synchronize()` everywhere (kills throughput).
- Mixing defaults and custom streams haphazardly (surprising waits). Use scoped contexts.

### 6.15 References

- MLX Streams (Python): usage, devices and streams, `mx.synchronize`
- MLX-Data: buffers, streams, and samples

---

## 7. Non-Square Orthogonality

Orthogonality isn't just for square matrices. This section documents robust patterns for semi-orthogonal matrices and
completion to a full orthonormal basis, using MLX and GPU-friendly kernels.

### 7.1 Definitions

- **Left-orthonormal (orthonormal columns):** Q ∈ R^{m×n}, m ≥ n, Q^T Q = I_n
- **Right-orthonormal (orthonormal rows):** Q ∈ R^{m×n}, n ≥ m, Q Q^T = I_m

### 7.2 Orthonormal Columns (Left)

```python
from metalfaiss.faissmlx.qr import pure_mlx_qr

def orthonormal_columns(X: mx.array) -> mx.array:
    # Two-pass MGS via MLX QR builds Q with Q^T Q = I
    Q, _ = pure_mlx_qr(X)
    return Q[:, : X.shape[1]]
```

### 7.3 Orthonormal Rows (Right)

```python
def orthonormal_rows(X: mx.array) -> mx.array:
    # Orthonormalize columns of X^T, then transpose back
    Qt, _ = pure_mlx_qr(mx.transpose(X))
    return mx.transpose(Qt[:, : X.shape[0]])
```

### 7.4 Completing to a Full Basis

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
        # two-pass MGS projection
        c1 = mx.matmul(mx.transpose(R), v)
        v  = v - mx.matmul(R, c1)
        c2 = mx.matmul(mx.transpose(R), v)
        v  = v - mx.matmul(R, c2)
        nrm = mx.sqrt(mx.sum(v*v))
        u = v / mx.where(nrm > 0, nrm, 1)
        R = mx.concatenate([R, u.reshape((m, 1))], axis=1)
    return R
```

### 7.5 GPU Notes

- Use the QR projection kernel (c = Q^T v) for large m,k to speed up re-orthonormalization.
- Consider HPC16x8 limb accumulation for projections and norms when drift appears.
- Random rotations for non-square transforms:
    - If d_in ≥ d_out: take first d_out columns of a left-orthonormal Q.
    - If d_out > d_in: build right-orthonormal rows and transpose.

---

## 8. Static Tuning Parameters

### 8.1 Goal

Keep production binaries fast and predictable by loading precomputed, per-device parameters (tile sizes, band sizes,
stream counts) rather than autotuning on user machines.

### 8.2 Where Params Live

A project config file such as `config/hardware_params.json` (format is up to you)

Example entries:

- metal / Apple M3 → AV=32×8, AT_B=8×32, QR dot mode=simd threshold, SVD band=32, streams=2
- metal / Apple M2 → AV=16×16, AT_B=16×16

### 8.3 How They Are Used

- A small loader (for example, `tuning.py`) reads the JSON and returns parameters for the current device (GPU via Metal
  or CPU).
- Kernels consult tuning first, then env overrides, then fallback heuristics.
- SVD and QR can also take band/streams and QR dot mode hints from the same file.

### 8.4 Override Order

1. Environment variables (e.g., `MLX_GEMM_TILE_AV=32x8`)
2. Static config (hardware_params.json)
3. Device detection heuristic (e.g., default to 16×16)

### 8.5 Regenerating Params

Use your project's benchmarking harnesses or microbenchmarks to explore shapes on your hardware:

- Tile sweep to pick GEMM tile sizes
- Bands/streams selection for SVD/QR and pipeline parallelism
- QR projection kernel mode/dot threshold exploration

Update the JSON entries with winners and keep rationale in commit messages.

### 8.6 Cross-Platform Note

macOS on Apple silicon does not support CUDA. If you target non-Apple systems with NVIDIA GPUs, you can extend this same
config format to include a "cuda" backend for those environments.

---

## 9. External Reference Map

This project's MLX + Metal kernel patterns are informed by a deeper body of work in the Ember ML repository. Those
curated docs contain real-world findings, pitfalls, and fixes that go beyond the official MLX documentation.

### 9.1 Primary Curated Path

`/Volumes/emberstuff/Projects/magentic-codex/codex-cli/agent_knowledgebase/mlx/`

### 9.2 Suggested Starting Points (High-Signal)

- `mlx.core.fast.metal_kernel.md` — API and call contract
- `linalg.md` — QR/SVD tiling notes, kernel wrapper patterns, sizing heuristics
- `docs_curated/HPC16x8.md` — Limb-based 128-bit accumulation for robust dot/norm on GPU
- `docs_curated/COMMON_PITFALLS.md` — Real causes of "expected expression", include collisions, and JIT churn
- `docs_curated/PORTING_FROM_PYTORCH.md` — RNG/key patterns; stateful to keyed conversion

### 9.3 Related Ember ML Code References (for Deeper Dives)

- `ember_ml/backend/mlx/linearalg/qr_ops.py` — Enhanced QR kernel with diagnostics/safety
- `ember_ml/backend/mlx/linearalg/cholesky_ops.py` — Single-thread and block-tiled Cholesky
- `ember_ml/backend/mlx/linearalg/svd_ops.py` — Power-iteration and tiling strategy
- `ember_ml/backend/mlx/linearalg/orthogonal_nonsquare.py` — Rectangular orthogonality + completion
- `ember_ml/backend/mlx/linearalg/eigen_ops.py` — Autoswitching patterns, HPC16x8 integration

### 9.4 Attribution

The kernel patterns and HPC techniques used here are adapted from Sydney Bach's Ember ML project (The Solace Project).
Where applicable, this repo mirrors those approaches and salutes the ingenuity that made them work on real hardware.

### 9.5 How This Repo Applies the Patterns

- `docs/mlx_reference/Comprehensive-MLX-Metal-Guide.md` — The definitive guide to writing, launching, and optimizing
  kernels with MLX in this project.
- `docs/research/Journal.md` — A log of experiments, benchmarks, and design rationale for our kernels.
- `docs/mlx_reference/Kernel-Guide.md` — Working kernel snippets (body+header), grid/tg selection, autoswitch ideas.
- `docs/mlx_reference/Orthogonality.md` — Practical left/right orthogonality and completion.
- `python/metalfaiss/faissmlx/kernels/qr_kernels.py` — Body-only projection kernel; header for includes.
- `python/metalfaiss/faissmlx/qr.py` — Two-pass MGS QR with optional kernel projections.
- `python/metalfaiss/faissmlx/svd.py` — MLX tiled subspace SVD; designed to slot in a kernelized Z-step.

If you're authoring new kernels, scan the curated docs first — they save days of guesswork by showing what actually
compiles and runs fast on Apple GPUs.

---

## Final Notes

This comprehensive guide consolidates all MLX + Metal reference materials for the xLSTM project. Each section preserves
detailed examples, code snippets, and practical guidance from the original documents. Use this as your primary reference
when developing Metal kernels, optimizing performance, or troubleshooting issues with MLX on Apple GPUs.

For questions or contributions, refer to the project's main documentation or the research journal for experimental
findings and ongoing work.

