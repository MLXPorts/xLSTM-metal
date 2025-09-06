Repository Structure (High-level)

- `mlstm_kernels/torch/`
  - `backend_module.py`: mLSTM backend wrapper and config (routes chunkwise/sequence/step).
  - `chunkwise/`: chunkwise (prefill) kernels and drivers.
    - `native/`: pure-PyTorch eager.
    - `native_compiled/`: pure-PyTorch compiled variants.
    - `queued_compiled/`: GPU-only queued compiled-step driver for MPS.
    - `metal/` and `triton_*` folders: legacy paths (Metal custom and Triton; disabled on Apple in this setup).
  - `recurrent/`: sequence/step kernels.
    - `metal/compiled.py`: compiled mLSTM step used for Apple MPS.
    - `native_sequence/`: pure-PyTorch sequence loop.
  - `parallel/`: fully-parallel (quadratic) kernels for analysis/experiments.

- `xlstm_official_full/`
  - `xlstm_large/`: standalone xLSTM Large model and config (requires `mlstm_kernels`).
  - `blocks/slstm/`: sLSTM implementation; includes compiled backend for MPS.

- `scripts/`
  - `runners/`: run/train/infer entrypoints (Ray default chunkwise backend).
    - `run_local_xlstm_mps.py`: run local HF checkpoint on MPS.
    - `run_hf_xlstm_metal.py`: run HF model id on MPS.
  - `benchmarks/`: throughput/latency harnesses.
  - `downloads/`: checkpoint acquisition.
  - `debug/`, `checks/`, `build/`, `experiments/`.

- `implementations/`
  - `pytorch/`, `metal/`, `mlx/`: organized legacy/reference implementations (top‑level import shims preserved).

- `research_archive/`
  - Archived experiments, notes, and demos; not wired into production runners.

Notes
- Root-level `xlstm_*.py` files are legacy shims that re-export from `implementations/...` to avoid breaking older imports.
- See `docs/REPO_HYGIENE.md` for a production vs experiments overview.

- `tools/`
  - `test_metal_parity.py`: mLSTM parity checks.
  - `test_slstm_parity.py`: sLSTM parity checks.

Conventions
- Kernel registry keys (Ray is default chunkwise):
  - `chunkwise` backends: `"chunkwise--<variant>"`, e.g., `chunkwise--ray_compiled_steps` (default), `queued_compiled_steps` (legacy).
  - `sequence` backends: `"native_sequence__<variant>"`, e.g., `native_sequence__metal`.
  - `step` backends: `"<variant>"`, e.g., `metal`.
- Compiled backends are strict: no CPU fallback; raise on unsupported device.
