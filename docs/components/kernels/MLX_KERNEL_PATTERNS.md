# MLX Kernel Patterns (Cheat Sheet)

Practical guidance for writing fast, correct MLX custom Metal kernels.

## Launch Semantics
- MLX `grid = threads`, `threadgroup = per‑group size`. Use tile size for the threadgroup.
- `threadgroup_position_in_grid` encodes tile indices; `thread_position_in_threadgroup` encodes intra‑tile indices.

## Threadgroup Memory (Tiles)
- Treat as a software‑managed cache.
- Cooperative **unique‑writer** loads: map each thread to a unique tile element.
- Two barriers per K‑tile iteration:
  - After loads (ensure visibility of shared stores)
  - After accumulation (before overwriting tiles)
- Use `threadgroup_barrier(mem_flags::mem_threadgroup)`; no divergence around barriers.

## Access Patterns
- Favor row‑major tiles and coalesced loads/stores.
- Inner loop iterates the shared dimension (`fma(tileA[row][p], tileB[p][col], acc)`).
- Optional: add `+1` padding on the second tile dimension to break conflict patterns.

## Tiling & Sizing
- Keep threadgroup size ≤ 1024 and each axis ≤ device limits.
- Optionally align square tile size `T` to `threadExecutionWidth`.
- Consider rectangular tiles if profiling shows wins, but keep correctness tests.

## Optional Optimizations
- Double buffering: ping‑pong shared tiles so loads for tile `t+1` overlap compute on tile `t`.
- Vectorized loads (e.g., `float4`) when alignment and shape permit to reduce instruction count.

## Validation
- Compare kernels against reference `mx.matmul` (≤1e‑4 tolerance for float32) across awkward shapes and partial tiles.
- Warm up kernels to populate caches, then record medians.

For concrete examples, see `mlx_fast_kernels/gemm_kernels.py` and `mlx_fast_kernels/qr_kernels.py`.
