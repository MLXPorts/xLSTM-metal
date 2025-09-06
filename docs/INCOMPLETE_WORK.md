# Open Work — Tracking List (2025-08)

The core MPS inference path is working well using compiled step and sequence kernels on MPS plus our chunkwise schedulers. This file tracks targeted enhancements and nice‑to‑haves; nothing here blocks usage.

## Improvements (Priority)
1. Per‑backend metrics and tracing
   - Add lightweight timing hooks in `queued_compiled` and `ray_compiled` drivers (per band/segment) and surface CSV/JSON.
   - Ensure `--stats-log/--stats-every` from `scripts/run_local_xlstm_mps.py` covers both prefill and decode phases.

2. Streams ergonomics (MPS)
   - Detect `torch.mps.Stream` support and auto‑tune stream count vs workers; avoid oversubscription.
   - Guarded enablement tied to PyTorch version.

3. Native compiled chunkwise parity
   - Keep `native_compiled_autograd` up to date with numerical stabilizations used in the compiled step path.
   - Expand small shape test harness to compare step vs chunkwise outputs on random seeds.

## Nice‑to‑Haves
1. Ray backend refinements
   - Actor pool sizing heuristics; optional placement groups (if we ever run multi‑host).
   - Better backpressure and heartbeat logging in local_mode.

2. API polish
   - Consolidate backend selection into a small helper; consistent error messages when MPS is unavailable.
   - Document env var → config mapping in a single place (README + TUNING_GUIDE).

3. Documentation debt
   - Keep this file and `HANDOFF_NOTES.md` in sync when adding/removing backends.
   - Add a short design rationale section (why bands × small chunks) with diagrams.

## Current Facts (for quick orientation)
- Step kernel: `metal` is a compiled PyTorch function executing on MPS with float32 math and stabilized gating; not handwritten Metal.
- Sequence kernel: `native_sequence__metal` runs decode by looping the compiled step.
- Chunkwise prefill:
  - `chunkwise--queued_compiled_steps`: thread pool queues many small step kernels (bands × chunks); all math stays on GPU.
  - `chunkwise--ray_compiled_steps`: Ray actors coordinate the same compiled step path.
  - `chunkwise--native_compiled_autograd`: compiled chunkwise path (inductor) as comparator.
- Entrypoint script: `scripts/run_local_xlstm_mps.py` — use `--chunkwise-backend`, `--chunk-size`, `--workers`, `--heads-per-band`, `--streams`.

## Out of Scope
- Handwritten Metal shader implementations for inference are not required for current performance goals.
- HPC limb arithmetic is unnecessary for the present numerics; revisit only if we push extreme sequence lengths/precision bounds.
