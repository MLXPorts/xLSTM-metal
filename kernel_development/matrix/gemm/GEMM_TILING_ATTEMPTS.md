# Archived Notes: Early GEMM Tiling Attempts

Summary

- We initially experimented with mixed tile shapes and treated `grid` as the
  number of threadgroups rather than the total number of threads (MLX uses
  dispatchThreads semantics). This under-dispatched work and broke indexing.
- Some early AT_B loaders allowed multiple threads to write the same
  threadgroup tile address, causing races and incorrect results.

What changed

- Grid now specifies total threads, and threadgroup is the per-group size.
  We round grid up to full tiles so `threadgroup_position_in_grid` enumerates
  tile indices, and `thread_position_in_threadgroup` covers the local tile.
- AT_B loaders assign unique writers per tile cell (row/col) and synchronize
  before consumption. Accumulation loops run over the shared-dimension slice.
- Row-major addressing is used consistently with `ensure_row_contiguous=True`.

Takeaways

- Always use grid = threads and threadgroup = tile when using
  `mx.fast.metal_kernel`.
- Avoid data races in shared tiles; one writer per address, fence before use.
- Prefer 2D grid math over runtime `/` and `%` in hot loops.
