Lab: Atomic MLX/Metal Validation

Purpose
- Provide small, focused experiments to validate proposed MLX improvements before landing code changes.
- Each script runs independently and prints correctness and throughput indicators.

Scripts
- mlx_headlinear_vs_blocklinear.py: Compare correctness and cost of block‑diagonal linear vs per‑head linear (no block matrix).
- mlx_multihead_layernorm_parity.py: Validate head‑aware normalization behavior vs naive GroupNorm.
- mlx_softcap_bench.py: Soft‑cap numeric parity and performance (pure MLX vs Metal kernel variant).
- gemm_tile_bench.py: Sweep GEMM tiles (16x16, 32x8, 8x32, 16x8, 8x16) for AV and AT_B on random sizes.
- mlx_sequence_precompute_scan_demo.py: Show dispatch savings by precomputing sequence projections.
- mx_streams_overlap_demo.py: Demonstrate MLX stream overlap and stream‑scoped synchronization.
- wwdc_checklist.md: WWDC patterns mapped to this repo’s kernels and lab checks.

Run
- All scripts are standalone: `python lab/<script>.py` (requires MLX installed).
- Some prints include timing; run multiple times for stable medians.

