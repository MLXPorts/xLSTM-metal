# Changelog

All notable changes to this project are documented here.

## [0.2.0] - 2025-09-01

- Ray compiled prefill backend: add optional asyncio actor with concurrency groups; optional Ray Compiled Graphs (beta) for steady-state chunk execution; per‑chunk prefill tok/s telemetry via gauges.
- xltop: add decode stats sparkline rendering from CSV tails; improve non‑curses mode parity with TUI.
- Memory watchdog: integrate across Ray workers via runtime env; keep GPU‑only behavior with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Optimizer: parameter sweep harness (`scripts/optimize_mps.py`) gains JSONL/CSV telemetry, prompt‑file support, GA mode, and plotting utility.
- Runner: `scripts/run_local_xlstm_mps.py` improved defaults for MPS compiled backends and Ray local‑mode; CLI gains dashboard/keep‑alive and memory logging flags.
- Experiments: add variable quantization Ray demo with compiled DAG option; ensure proper `ray.shutdown()` on exit.
- Docs: expand Agents guide (operations, safety, cleanup), Apple MPS tuning notes, and monitoring/observability guidance.

Notes
- Default chunkwise backend remains `chunkwise--ray_compiled_steps` in local_mode=1.
- For multi‑process Ray (dashboard), actors are terminated and Ray auto‑shutdowns unless `XLSTM_RAY_AUTOSHUTDOWN=0`.

[0.2.0]: https://github.com/your-org/xlstm/releases/tag/v0.2.0

