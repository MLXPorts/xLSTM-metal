Experiment: 2025‑09‑10‑mlx‑research (MLX xLSTM)

Goal
- Explore MLX (Metal) performance and stability improvements for xLSTM with lab-first prototypes, measurements, and reports before integration.

Contents
- Scripts: benches and parity checks for soft caps, per‑head linear, head‑aware LN, GEMM tiling, sequence precompute, smoke and parity tests.
- Kernels: experimental MLX fast Metal kernel for multi‑head layernorm (SIMD-group implementation).
- Docs: plan, lab report, journal (provenance and status).

How to run
- All scripts are standalone. Example:
  - `PYTHONPATH=. python lab/2025-09-10-mlx-research/mlx_softcap_bench.py`
  - `PYTHONPATH=. python lab/2025-09-10-mlx-research/gemm_tile_bench.py`
  - `PYTHONPATH=. python lab/2025-09-10-mlx-research/mhln_bench.py`

Notes
- Keep experiments self‑contained in this folder.
- Only promote techniques into production after lab validation and documentation.
