# runners

User-facing entrypoints to run, generate, and train.

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

Quick Start (zero flags)
- `PYTHONPATH=. python scripts/runners/xlstm_quick.py --prompt "Hello" --new 16`
- Auto‑discovers `./xlstm_7b_model` (or the first subdir containing `config.json`).
- Uses Ray (local_mode) + compiled MPS backends by default.

Typical
- `run_local_xlstm_mps.py` (already in scripts/): main entry with CLI and env mapping for MPS backends.
- `run_xlstm.py` (moved here): minimal example that loads a local HF checkpoint and runs generate.
- `inference.py`: small harness for forward/generate in custom experiments.
- `train_xlstm.py`: training scaffold (if used in your workflow).

Calling graph
`run_local_xlstm_mps.py` → `xlstm_official_full.xlstm_large` (model) → `mlstm_kernels` backends (step=metal compiled on MPS; sequence=native_sequence__metal; chunkwise={ray,queued}).
