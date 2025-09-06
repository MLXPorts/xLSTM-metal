<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

WWDC16-Inspired Optimization Patterns For MLX + Metal

This guide translates concrete shader optimization patterns from Apple's “Advanced Metal Shader Optimization” (WWDC16 Session 606) into practices applicable to MLX’s `mx.fast.metal_kernel`. It focuses on patterns for our math kernels (QR/SVD/GEMM) and highlights how MLX's abstractions map to low-level Metal concepts.

1) Address Spaces And “Constant Buffer” Ideas

- Device vs constant:
  - Metal offers `device` (read/write) and `constant` (read‑only, cached) address spaces. Small, read‑only data (e.g., shapes/flags) belongs in `constant` for preloading and reuse.
  - MLX binds your inputs as buffers under the hood; you don’t control qualifiers from Python. Practical adaptation:
    - Pack tiny params (shape, flags, eps) in small arrays and load them once into registers at the top of the kernel body.
    - Example (inside kernel source):
      `int m = int(shape[0]), n = int(shape[1]), k = int(shape[2]);`
    - Keep access statically bounded and avoid pointer chasing.

2) Compute Kernel Organization (Amortize Launch Overhead)

- Do enough work per thread:
  - Process 2 outputs per thread (e.g., two columns in GEMM) if register pressure allows.
  - Reuse loaded tiles: accumulate into multiple accumulators `acc0, acc1`.
  - Trade‑off: more registers lowers occupancy; measure before committing.
- Split phases across kernels instead of global barriers:
  - No device‑wide barrier in MSL. Use two kernels (e.g., dot then update for QR; A@V then Aᵀ@B for SVD) for clear sync points and simpler tuning.

3) Barriers: Use The Smallest Scope

- Prefer `threadgroup_barrier(mem_flags::mem_threadgroup)` when sharing TG arrays; use `mem_device` only when reading/writing global buffers across phases.
- If you can constrain to a single warp/group, `simdgroup_barrier` can be cheaper. In practice we tile with 16×16 TGs and use TG barriers.

4) Data Types (Register Footprint, ALU Throughput)

- Apple GPUs use 16‑bit register units; smaller types can improve occupancy:
  - Consider `half` for intermediate math that tolerates reduced precision; keep accumulators in `float`.
  - Use `ushort` for local IDs where appropriate; we generally keep indices as `int` for safe addressing.
  - Avoid mixing `half` with float literals (`2.0`); use `half` literals (`2.0h`).

5) Arithmetic (Built‑ins, Fast‑Math)

- Fast‑math is on by default; take advantage of:
  - `fma(a,b,c)` (fused multiply‑add) — we adopted this in GEMM tiles and QR update.
  - Built‑ins like `abs`, `saturate` are free modifiers; prefer them to manual code.
- Integer division/modulus:
  - Avoid divides by non‑constants in hot loops. Precompute reciprocals and multiply, replace /2ⁿ with shifts. MLX doesn’t expose function constants; prefer arithmetic transforms.

6) Control Flow (Uniform vs Divergent)

- Prefer uniform control flow across a warp; divergent branches serialize.
- Use ternary (select) for fast branchless decisions:
  - `x = cond ? a : b;`
  - Avoid “multiply by 0/1” tricks.

7) Memory Access (Vectorization, Stack, Addressing)

- Coalesce loads/stores; stage tiles in TG memory.
- Arrange structs for vectorizable access (SoA > AoS in many kernels).
- Avoid dynamically‑indexed, non‑constant stack arrays. The performance cost can be **catastrophic**. The session noted a real-world app that lost **30% of its performance** due to a single 32-byte dynamically indexed array. Compilers may unroll fixed-size loops to eliminate this, but it's a major pitfall to avoid.
- Use `int` (or smaller) for device memory addressing; prefer signed `int` over `uint` in index math to avoid extra instructions.

8) Latency, Occupancy, And Hiding

- Threadgroup memory and registers cap occupancy. Keep TG arrays modest and avoid excessive accumulators.
- Interleave independent work to hide latency if texture/long ops appear; for our math kernels, tiling + FMA dominates.

9) Putting It Into Practice (Our Kernels)

- GEMM tiles (A@V; Aᵀ@B):
  - 16×16 tiles; TG arrays for A and V/B; barriers between load/accumulate; explicit `fma` in inner loop; `int` indices for addressing.
  - Try 32×8 or 8×32 on your device.
- QR helpers:
  - Separate kernels for dot and update; both use `fma`; leave projection norms to MLX unless reductions are a bottleneck.
- SVD Z‑step:
  - Two GEMM‑like kernels beat a monolith for maintainability and tuning. Banded processing can reduce peak memory; streams only help at large sizes; benchmark first.

10) Checklists
Before optimizing, ask:
  - Are loads coalesced? Are you staging into TG memory?
  - Is there an obvious `fma` opportunity?
  - Is integer division avoidable? Can you precompute factors?
  - Are your barriers the smallest scope that works?
  - Are per‑thread workloads large enough to amortize launch overhead without crushing occupancy?
  - Are you using the smallest practical data types (`half`, `short`)?
  - Have you eliminated dynamically-indexed stack arrays?

References

- Apple WWDC16 Session 606: “Advanced Metal Shader Optimization” (address spaces, preloading, kernel organization, data types, control flow, memory access, occupancy)
- Our docs: Comprehensive-MLX-Metal-Guide.md, Metal-Primer.md, Shader-Optimization-Tips.md
