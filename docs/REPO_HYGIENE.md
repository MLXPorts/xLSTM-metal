Repository Hygiene: Production vs Experiments

Purpose: make it easy to find the established, supported code paths and avoid confusion with experiments or legacy references.

Production (supported)
- `scripts/`: main entrypoints and benchmark runners
  - `scripts/run_local_xlstm_mps.py` (MPS/Ray/queued backends)
  - `scripts/run_local_xlstm_mlx.py` (pure MLX path)
  - `scripts/benchmarks/run_mlx_benchmarks.py` (MLX benchmarks)
- `mlx_fast_kernels/`: MLX kernels (GEMM/QR/SVD/IVF) used by the fast head and utilities
- `mlstm_kernels/`: PyTorch kernels, compiled step/sequence + memory watchdog and utilities
- `tools/`: runtime config, tuning, streams, telemetry helpers
- `docs/`: current guides (implementation, tuning, runtime config, benchmarks, testing)
- `tests/`: focused tests including MLX parity, GEMM/QR/SVD/IVF, and xLSTM MLX inference parity
- `configs/`: device defaults and runtime presets

Reference (vendored or upstream snapshots)
- `implementations/`: organized legacy and reference implementations (pytorch/metal/mlx)
- `xlstm_official/`, `xlstm_official_full/`: upstream-style layouts for study and comparison
- `pytorch_implementation/`, `mlx_implementation/`: third-party/reference projects kept for context

Legacy shims (kept for backward compatibility)
- Root-level modules like `xlstm_mlx.py`, `xlstm_pytorch.py`, `xlstm_metal_optimized.py`, etc. They simply re-export from `implementations/...` to avoid breaking older imports in scripts/tests. Prefer importing from `implementations/...` directly.

Archived experiments
- `research_archive/`: experiments, naive baselines, notes, and demos moved out of the main path
  - `naive_svd_at_a_v.py`, `GEMM_TILING_ATTEMPTS.md`, `pytorch_metal_kernels_demo.py`
  - `legacy_shims/`: formerly root import shims (unused in-repo) moved here to reduce clutter

Root shims status
- Kept (compat): `xlstm_mlx.py`, `xlstm_pytorch.py`, `xlstm_pytorch_inference.py`, `xlstm_metal_complete.py` (emit DeprecationWarning)
- Archived: `xlstm_chunked_parallel.py`, `xlstm_jit_metal.py`, `xlstm_jit_simple.py`,
  `xlstm_metal_hpc_limb.py`, `xlstm_metal_hpc_limb_fixed.py`, `xlstm_metal_kernels.py`,
  `xlstm_metal_optimized.py`, `xlstm_pytorch_enhanced.py`, `xlstm_streaming_inference.py`,
  `xlstm_torch_compile_fixed.py`, `xlstm_unified_metal.py`

Conventions
- Prefer programmatic runtime config over environment variables (see `docs/MLX_RUNTIME_CONFIG.md`).
- Keep new experiments under `lab/` or `research_archive/` with a brief README and do not wire them into production runners unless promoted.
- No Swift port: this repository does not contain an MLX Swift implementation. Any mentions of Swift in docs are historical/contextual and have been clarified.
- No symlinks/hardlinks: repository content must be regular files/directories. The policy check `scripts/lint/check_repo_policy.py` errors on symlinks and hardlinks (except under large artifact caches like `model_cache/`, `runs/`, `outputs/`, `quarantine/`, `xlstm_7b_model/`).
- No submodules: nested Git repositories are not allowed. If you need to vendor code, include it as regular files (no nested `.git`). The policy check flags nested `.git` directories.
