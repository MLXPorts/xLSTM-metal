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

Guidelines
- Use the production implementations under `mlx_fast_kernels/` for speed.
- Treat archive items as educational references or benchmarks, not as active
  building blocks.
