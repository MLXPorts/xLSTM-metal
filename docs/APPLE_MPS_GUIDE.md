Apple/MPS Guide: Running xLSTM on Apple Silicon (GPU-only)

Overview
- This repo provides pure-PyTorch compiled backends for mLSTM and sLSTM on Apple Silicon (MPS), without Triton.
- All math runs on GPU. Compiled kernels are strict: if compilation fails, they raise (no CPU fallback).

Backends (Apple Defaults)
- mLSTM step: `step_kernel="metal"` (compiled step)
- mLSTM sequence: `sequence_kernel="native_sequence__metal"` (compiled loop over step)
- mLSTM chunkwise (prefill): `chunkwise_kernel="chunkwise--queued_compiled_steps"` (GPU-only queued compiled-step driver)
- sLSTM: backend switches to `native_compiled` automatically on MPS (strict compile)

Quick Start (Local Checkpoint)
- Command:
  - `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 \
     python scripts/run_local_xlstm_mps.py \
       --model_path /path/to/xlstm_7b_model \
       --prompt "The capital of France is" \
       --max_new_tokens 64 \
       --workers 6 --heads-per-band 4`

Tuning Knobs (Environment)
- `XLSTM_MPS_WORKERS` (default 6): CPU coordinator threads.
- `XLSTM_MPS_HEADS_PER_BAND` (default 4): heads per task.
- `XLSTM_MPS_STREAMS` (default = workers): dedicated MPS streams (one per worker is typical).
- `XLSTM_MPS_AUTOSCALE=1`: enable a micro-probe to adjust heads per band automatically.
- `XLSTM_MPS_WARMUP=0|1` (default 1): enable step-kernel warm-up.
- `PYTORCH_ENABLE_MPS_FALLBACK=0`: enforce GPU-only execution.

Memory watchdog (unified memory safety)
- `XLSTM_MEM_WATCHDOG=1` (default): enable watchdog thread during prefill.
- `XLSTM_MEM_SOFT_PCT` / `XLSTM_MEM_HARD_PCT`: soft/hard process-RSS thresholds as fraction of total RAM (defaults 0.85 / 0.92).
- `XLSTM_MEM_SOFT_MB` / `XLSTM_MEM_HARD_MB`: absolute MB thresholds (override pct if set).
- `XLSTM_MEM_POLL_MS` (default 200): sampling period.
- Action on soft threshold: warns, empties MPS cache, halves `chunk_size` (down to `XLSTM_MIN_CHUNK`, default 8).
- On hard threshold: aborts cleanly with a helpful error.
- `XLSTM_MEM_ACTION`: soft-limit actions list (e.g., `warn` or `warn,empty_cache`).
- `XLSTM_MIN_CHUNK`: clamp lower bound for chunk shrinking (default 8).
- `XLSTM_SHRINK_ON_SOFT=0` disables chunk shrinking entirely.
- Optional logs: `TORCH_LOGS=+dynamo` and `TORCHDYNAMO_VERBOSE=1` for compile debugging.

Notes on Chunkwise Prefill
- Fully compiled chunkwise can hit Metal’s per-kernel argument limits on long sequences in the current prototype MPS compiler.
- The `queued_compiled_steps` backend sidesteps this by dispatching many small compiled step kernels across head bands and sequence tiles, keeping the GPU saturated without hitting kernel-arg limits.

Pseudo‑Kernel Fusion (torch.compile)
- We use `torch.compile` to treat the xLSTM step as a fused “pseudo‑kernel”. Per‑timestep math (gates; (C,N,M) update; readout) is composed inside the compiled region to reduce launches and memory traffic.
- Inner tiling (`T_inner`): instead of a scalar one‑step compile, we unroll a small block of steps (e.g., 4 or 8) to enlarge fusion windows. Drivers loop over these blocks to cover the logical `chunk_size` without changing semantics.
- Canon semantics: logical `chunk_size` remains fixed; time order is strict; (C,N,M) flows exactly across sub‑tiles and chunk boundaries. Runtime chunk shrinking is non‑canonical and should remain off unless used as an OOM escape hatch on UMA.
- See also: PYTORCH_MPS_FUSION_NOTES.md for deeper design notes.

Ray lifecycle (local vs daemon)
- Default is in‑process (`XLSTM_RAY_LOCAL_MODE=1`), which avoids starting external Ray daemons.
- If you set `XLSTM_RAY_LOCAL_MODE=0`, Ray will launch a local head (raylet/GCS). We auto‑shutdown at the end of the run (`XLSTM_RAY_AUTOSHUTDOWN=1`, default), but orphaned sessions can linger if a process crashes. Use `ray stop --force` to clean up.

Ray Dashboard
- Enable via runner: `--ray-dashboard [--ray-dashboard-port 8265]` to start the web UI at `http://127.0.0.1:8265`.
- Keep it up after the run with `--ray-keep-alive`; we terminate actors first to free MPS memory, then leave the head up.
- Note: dashboard requires `XLSTM_RAY_LOCAL_MODE=0` (multi‑process). This can duplicate GPU allocations across actors if you pass large MPS tensors to multiple actors. Our Ray backend limits this by terminating actors promptly; for safest memory usage, prefer queued backend when using the dashboard just for observability.

Hugging Face Path (Downloads Model)
- `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_hf_xlstm_metal.py`
- Uses the Apple defaults for step/sequence/chunkwise.

Validation
- mLSTM parity: `PYTHONPATH=. python tools/test_metal_parity.py`
- sLSTM parity: `PYTHONPATH=. python tools/test_slstm_parity.py`

Metrics & Telemetry
- Local runner prints total time; queued backend also tracks aggregate steps/time internally.
- You can estimate prefill throughput by: steps/time from backend, or simply tokens/sec from CLI output.

High‑Memory Aggressive Profile (256 GB UMA)
- Goal: prioritize throughput; only warn near the cliff, no chunk shrinking.
- Environment:
  - `XLSTM_MEM_ACTION=warn`
  - `XLSTM_MEM_SOFT_MB=215000` (≈215 GB), `XLSTM_MEM_HARD_MB=235000` (≈235 GB)
  - `XLSTM_SHRINK_ON_SOFT=0`
  - Set `--chunk-size 64` or higher; keep `--heads-per-band 4` (try 2 as well).
- Example run:
  - `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 \
     python scripts/run_local_xlstm_mps.py --model_path /path/to/xlstm_7b_model \
       --chunkwise-backend ray_compiled_steps --chunk-size 64 --heads-per-band 4 \
       --prompt "…" --max_new_tokens 32 \
       --mem-soft-mb 215000 --mem-hard-mb 235000 --mem-action warn --no-mem-shrink`
