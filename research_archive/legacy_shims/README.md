Legacy Shim Modules (Archived)

These modules previously lived at repo root as import shims to avoid breaking
old code (e.g., `from xlstm_streaming_inference import ...`). They now live
under `research_archive/legacy_shims/` to reduce top-level clutter.

Preferred imports
- Use the organized paths under `implementations/` instead, for example:
  - `from implementations.pytorch.xlstm_streaming_inference import ...`
  - `from implementations.metal.xlstm_metal_optimized import ...`

Kept at root (for compatibility)
- `xlstm_mlx.py`, `xlstm_pytorch.py`, `xlstm_pytorch_inference.py`,
  `xlstm_metal_complete.py` — these still have active in-repo users and
  emit a DeprecationWarning on import.

