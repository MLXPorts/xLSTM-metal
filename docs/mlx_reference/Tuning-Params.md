<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

Static Tuning Parameters (No Autotune in Production)

Goal
- Keep production binaries fast and predictable by loading precomputed, per‑device parameters (tile sizes, band sizes, stream counts) rather than autotuning on user machines.

Where params live
- `python/metalfaiss/faissmlx/config/hardware_params.json`
  - Example entries:
    - metal / Apple M3 → AV=32×8, AT_B=8×32, QR dot mode=simd threshold, SVD band=32, streams=2
    - metal / Apple M2 → AV=16×16, AT_B=16×16
    - cuda / sm_80 → AV=32×8, AT_B=8×32

How they are used
- `faissmlx/tuning.py` loads the JSON and returns parameters for the current backend (Metal or CUDA) and device.
- Kernels (`gemm_kernels.py`) consult tuning first, then env overrides, then fallback heuristics.
- SVD and QR can also take band/streams and QR dot mode hints from the same file.

Override order
1) Environment variables (e.g., `METALFAISS_GEMM_TILE_AV=32x8`)
2) Static config (hardware_params.json)
3) Device detection heuristic (e.g., default to 16×16)

Regenerating params
- Use the bench harnesses to explore shapes on your hardware:
  - Tile sweep: `python -m python.metalfaiss.unittest.test_kernel_autotune_bench -q`
  - Bands/streams: `python -m python.metalfaiss.unittest.test_band_streams_bench -q`
  - QR projection: `python -m python.metalfaiss.unittest.test_qr_proj_bench -q`
- Update the JSON entries with winners and keep comments in commit messages.

CUDA notes
- `tuning.py` probes `mlx.core.cuda.device_info()` when available; populate entries keyed by compute capability (e.g., `sm_80`, `sm_90`) or names.
- For hybrid Apple+CUDA deployments, entries for both backends can live side‑by‑side in the same file.

