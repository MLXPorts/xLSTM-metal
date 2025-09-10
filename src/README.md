# implementations

Organized xLSTM reference and experimental implementations.

Subfolders
- `pytorch/` — PyTorch reference and enhanced variants (streaming, chunked, compile-fixed).
- `metal/` — Metal/MPS-focused experimental variants and unified Metal paths.
- `mlx/` — Apple MLX variant(s) and related helpers (see also `mlx_implementation/`).

Notes
- The primary production path uses `xlstm_official_full` + `mlstm_kernels` with compiled MPS backends.
- These files are maintained for research, benchmarking, and comparison.

