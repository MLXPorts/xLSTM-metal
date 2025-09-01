# Agents Guide: Operating xLSTM locally with Codex CLI

This document is for human operators and automation agents working on this repo. It explains what the project is, which tools and scripts are available, how to run them safely on Apple Silicon, and how to observe/clean up resources. All commands assume macOS on Apple Silicon, conda env `base` (Python 3.12), and that you run with `PYTHONPATH=.`.

## What This Project Is
- Goal: High‑throughput xLSTM (mLSTM + sLSTM) inference on Apple Silicon GPUs via compiled PyTorch MPS backends.
- Core idea: Keep kernels small but compiled, schedule them efficiently across head‑bands and sequence tiles, and do all math on GPU.
- Prefill (long context) is chunked and scheduled by one of:
  - `chunkwise--ray_compiled_steps`: Ray actors coordinate many compiled step kernels (default).
  - `chunkwise--queued_compiled_steps`: CPU thread pool schedules compiled step kernels.
  - `native_compiled_autograd`: fully compiled native chunkwise (experimental on MPS; can hit Metal arg limits).
- Decode: compiled step kernel in a loop (`step_kernel="metal"`; `sequence_kernel="native_sequence__metal"`).

## Quick Start
- Local checkpoint run (Ray backend, dashboard optional):
  - `conda run -n base python scripts/run_local_xlstm_mps.py \
      --model_path ./xlstm_7b_model \
      --prompt "The capital of France is" --max_new_tokens 32 \
      --chunkwise-backend ray_compiled_steps \
      --chunk-size 64 --heads-per-band 4 \
      --ray-dashboard --ray-keep-alive`
- Enforce GPU‑only (recommended): set `PYTORCH_ENABLE_MPS_FALLBACK=0` in your environment.

## Tools At A Glance
- `scripts/run_local_xlstm_mps.py`: main entry for local inference
  - Backends: `--chunkwise-backend {ray_compiled_steps,queued_compiled_steps,native_compiled_autograd}`
  - Tuning: `--chunk-size`, `--heads-per-band`, `--workers` (queued)
  - Memory watchdog/telemetry: `--mem-log`, `--mem-every`, `--mem-soft-pct|--mem-hard-pct`, `--mem-soft-mb|--mem-hard-mb`, `--mem-action`
  - Ray/Dashboard: `--ray-dashboard`, `--ray-dashboard-port`, `--ray-local-mode {0,1}`, `--ray-keep-alive`
- `scripts/optimize_mps.py`: random/GA sweeps for prefill/decode throughput (writes `runs/mps_opt/<run>/`)
- `scripts/xltop.py`: terminal monitor with TUI and machine‑friendly modes
  - Curses UI: `conda run -n base python scripts/xltop.py`
  - Polling “top”: `conda run -n base python scripts/xltop.py --no-curses --poll 1.0`
  - JSON one‑shot: `conda run -n base python scripts/xltop.py --json`
  - NDJSON stream: `conda run -n base python scripts/xltop.py --json-stream --poll 0.7 --count 10`
  - Stdin control (for agents): `--stdin-commands` then send lines: `kill <pid>`, `ray stop`, `empty_cache`, `interval <sec>`, `quit`
- Ray CLI (installed in `base`): `ray status`, `ray list actors`, `ray memory`, `ray stop --force`

## Memory Watchdog & Telemetry
- Module: `mlstm_kernels/torch/monitoring/memory.py`
  - Samples process RSS, system available/total, and `torch.mps` allocated/reserved.
  - Optional CSV logging: `--mem-log` (runner), or use `MemoryMonitor(log_csv_path=...)` directly.
  - Thresholds: percentage of total or absolute MB; soft actions and hard abort.
- Defaults (tuned for safety; override for high‑capacity UMA):
  - Soft 85%, hard 92% of RAM; poll 200 ms; soft actions: `warn,empty_cache`.
- Aggressive high‑memory profile (256 GB UMA):
  - `XLSTM_MEM_ACTION=warn` and absolute caps like `XLSTM_MEM_SOFT_MB=215000`, `XLSTM_MEM_HARD_MB=235000`.

## Ray Lifecycle & Dashboard
- Local‑mode by default: `XLSTM_RAY_LOCAL_MODE=1` keeps execution in‑process (no daemons).
- Dashboard mode:
  - Enable with runner flags: `--ray-dashboard [--ray-dashboard-port 8265]`. This implies `local_mode=0` and starts a local head.
  - Keep the head alive after the run with `--ray-keep-alive`. The driver will terminate actors first to free GPU memory.
  - Auto‑shutdown: by default `XLSTM_RAY_AUTOSHUTDOWN=1` and the backend calls `ray.shutdown()` in a `finally` block if it started Ray.
- Cleanup on crash: `ray stop --force`.

## Key Environment Variables
- Backends & scheduling:
  - `XLSTM_CHUNKWISE_BACKEND={ray_compiled_steps|queued_compiled_steps|native_compiled_autograd}`
  - `XLSTM_MPS_WORKERS` (queued), `XLSTM_MPS_HEADS_PER_BAND`, `XLSTM_MPS_STREAMS` (optional)
- Memory:
  - `XLSTM_MEM_WATCHDOG=1`, `XLSTM_MEM_POLL_MS`, `XLSTM_MEM_SOFT_PCT|HARD_PCT`, `XLSTM_MEM_SOFT_MB|HARD_MB`, `XLSTM_MEM_ACTION`
- Ray:
  - `XLSTM_RAY_LOCAL_MODE=1`, `XLSTM_RAY_DASHBOARD=1`, `XLSTM_RAY_DASHBOARD_PORT=8265`, `XLSTM_RAY_AUTOSHUTDOWN=1`
- PyTorch/MPS: `PYTORCH_ENABLE_MPS_FALLBACK=0`

## Common Workflows
- Run + observe with dashboard and xltop
  1) Start: `conda run -n base python scripts/run_local_xlstm_mps.py --model_path ./xlstm_7b_model --prompt "…" --max_new_tokens 32 --chunk-size 64 --heads-per-band 4 --ray-dashboard --ray-keep-alive --mem-log runs/mem.csv`
  2) Open dashboard: http://127.0.0.1:8265
  3) In another terminal: `conda run -n base python scripts/xltop.py --no-curses --json-stream --poll 1.0` (or full TUI)
  4) Cleanup when done: `conda run -n base ray stop --force`
- High‑throughput local run (no dashboard, in‑process Ray):
  - `conda run -n base python scripts/run_local_xlstm_mps.py --model_path ./xlstm_7b_model --chunk-size 64 --heads-per-band 4 --mem-action warn --mem-soft-mb 215000 --mem-hard-mb 235000`
- Parameter sweep
  - `conda run -n base python scripts/optimize_mps.py --backend ray --mode random --trials 40` → inspect `runs/mps_opt/<run>/summary.csv`

## Safety & Cleanup
- Prefer `XLSTM_RAY_LOCAL_MODE=1` when you don’t need the dashboard.
- When using multi‑process Ray (dashboard), rely on our auto‑shutdown. If anything is left over: `ray stop --force`.
- On memory pressure or thrashing, check:
  - `scripts/xltop.py --json` (look at `rss_mb`, `mps_alloc_mb`, and `mps_recommended_gb`)
  - `/usr/bin/vmmap -summary <pid>` (IOAccelerator resident footprint)
- If a run goes rogue: `kill -TERM <pid>` (and then `-KILL` if needed); xltop TUI has a `K` prompt to kill by PID.

## Glossary
- Prefill: processing the entire input prompt (sequence chunked across many compiled step kernels).
- Decode: token‑by‑token generation using the compiled step kernel.
- Heads‑per‑band: parallelism granularity across attention heads; controls the number of concurrent tasks/actors.
- Ray local mode: executes tasks/actors in the driver process, avoiding external Ray daemons.

## Notes for Agents (Automation)
- Always run Python through conda base: `conda run -n base python …`. Avoid `python3` from the system.
- Set `PYTHONPATH=.` when invoking scripts from repo root so imports resolve.
- For structured telemetry, prefer `xltop --json-stream` and parse NDJSON events.
- To minimize interference, use warn‑only watchdog (`XLSTM_MEM_ACTION=warn`). Runtime chunk-size shrinking has been removed to preserve canonical behavior.

### Local Policy Hooks
- Install pre-commit hooks once: `pip install pre-commit && pre-commit install`.
- Hooks enforce: no mocks in prod, no “simplified/toy/placeholder” wording, and `ray.shutdown()` presence when `ray.init()` is used.

---
For deeper tuning and background, see:
- `docs/APPLE_MPS_GUIDE.md`
- `docs/TUNING_GUIDE.md`
- `docs/LOGGING_AND_OBSERVABILITY.md`
- `docs/PYTORCH_MPS_FUSION_NOTES.md`
- `docs/ANE_DEPLOYMENT_GUIDE.md`
- `FINAL_IMPLEMENTATION_REPORT.md`
