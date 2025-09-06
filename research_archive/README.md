# Research Archive

This folder contains approaches and reference implementations that were
useful during exploration but are not part of the production path because
faster or simpler alternatives now exist. Keeping them here avoids confusing
future readers while preserving the ideas and code for reference.

Contents
- `naive_svd_at_a_v.py` — baseline MLX/Metal kernel computing Z = Aᵀ(A V)
  with a 1D launch and nested loops. Correct and simple, but slower than the
  tiled two‑kernel path (`gemm_av`, `gemm_at_b`).
- `GEMM_TILING_ATTEMPTS.md` — notes on early tiling attempts (grid semantics,
  non‑square tile loading, and addressing) and why they were replaced.
- `pytorch_metal_kernels_demo.py` — PyTorch + Metal experiment showing how a
  custom Metal kernel could be bridged into PyTorch. Educational; not used by
  production backends.

Legacy import shims (moved here if unused)
- See `research_archive/legacy_shims/README.md` for a list of archived shims
  and preferred `implementations/...` imports.

Guidelines
- Use the production implementations under `mlx_fast_kernels/` for speed.
- Treat archive items as educational references or benchmarks, not as active
  building blocks.

Moved from root
- `pytorch_metal_kernels_demo.py` (formerly `pytorch_metal_kernels.py`)
