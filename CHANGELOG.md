# Changelog

All notable changes to this project are documented here. Entries are derived from commit history, grouped by release, with a commit‑by‑commit digest. Items related to external projects removed from the repo are omitted.

## [0.2.0] - 2025-09-02

Highlights
- Ray compiled prefill: optional asyncio actors with concurrency groups; optional Ray Compiled Graphs (beta) for steady‑state chunk execution; prefill tok/s gauges.
- xltop: decode tok/s sparkline from CSV tails; better non‑curses parity.
- Optimizer: GA/random sweeps with JSONL/CSV telemetry; plotting utility.
- Memory watchdog: integrated across Ray workers; preserves GPU‑only (`PYTORCH_ENABLE_MPS_FALLBACK=0`).
- Runner: improved defaults, dashboard/keep‑alive flags, stats/mem logs.
- Experiments: variable quantization demo; clean Ray shutdown on exit.
- Docs/ops: expanded agents guide, tuning and observability notes; policy pre‑commit hook; ignore `runs/` outputs.

Commit digest
- b315fd7 release(v0.2.0): version bump, CHANGELOG, README quick start, Ray shutdown in varquant experiment, ignore runs/, pre‑commit policy hook; include Ray driver and xltop updates
- 8ea7c2e Add signal handling and stats tracking to xLSTM scripts
- a24adc8 Refactor code structure for improved readability and maintainability
- 2758c23 Add xLSTM‑7B model download and loading scripts
- 03290e1 Merge pull request #12 (feature/cfc‑hybrid)
- 6c7b6d0 Journal: add guide, index, first entries; journal helpers (render_summary)
- 63d2d4e Packaging: add pyproject.toml; make tools.{telem,harvest} importable; expose console scripts
- ff18c40 Checkpoint: docs + telemetry updates (casual)
- 48aecad Docs: add research journal, provenance; probes + tests
- 4106858 Telemetry: add trace_hash + TelemetryLogger; integrate into trainer/demo
- fea7887 Tighten bolts: shape/assert guards; GPU‑aware tests
- 48daaa1 Phase‑key fusion for MemoryCube keys; ponder trainer; wire times
- 918ed98 Docs: add CONTEXT_SEED and CHANGELOG; link from overview
  (former HRM-related notes removed)
- 57bb3b9 feature(cfc‑hybrid): CfC‑hybrid override in chunkwise drivers (queued/ray)
- ac200c7 feature(cfc‑hybrid): top‑k sparse bias option in CfC calibrator; CLI `--cfc-topk`
- 247969c feature(cfc‑hybrid): add CfC logit calibrator (experimental) and CLI flags
- 37fb9be Checkpoint: docs + monitoring + Ray hygiene + TUI + experiments (CfC + DCC + reversible)
- f769cac Docs(paper): scaffold outline/abstract/references; tools(plot): add plot_opt_results.py
- 6397c02 Docs(experiments): capture post‑optimization ideas (side‑channel, hierarchy tokens, multi‑timescale…)
- a6dae67 feat(mps/opt): add `--prompt-file`, seed/tag/logging to optimizer
- ebac42b feat(mps/opt): add observability to optimizer (JSONL + CSV + run meta)
- a19b50f Chore(checkpoint): summarize MPS xLSTM progress
- fe50d0b Docs(plan): formalize sLSTM + multi‑kernel scheduling roadmap
- 3f11728 feat(mps/opt): add optimizer script (random + GA) for backend parameters
- 267469a feat(mps/ray): Ray actor‑based chunkwise backend (local_mode=1; GPU‑only; warm‑up; registry)
- 7386977 feat(mps/queue): streams‑safe queued chunkwise; warm‑up; autoscale; tuning flags
- 7c285c4 Chore(repo): remove embedded repos used for reference only (skipped details)
- 4289f40 feat(mps): Apple/MPS compiled backends + queued chunkwise for xLSTM
- 7971cbb Implement PyTorch JIT + Metal integration for xLSTM
- bfde744 Research and implement PyTorch Metal kernel integration
- d2560bd Add complete Metal kernel xLSTM implementation
- 4915551 Add unified Metal‑optimized xLSTM implementation
- e7fecd0 Initial commit: xLSTM implementation suite

Compatibility notes
- Default chunkwise backend is `chunkwise--ray_compiled_steps` (Ray local‑mode by default).
- With dashboard mode, actors are cleaned up and Ray auto‑shutdowns unless `XLSTM_RAY_AUTOSHUTDOWN=0`.

## [0.1.0] - 2025-08 (inferred)

Highlights
- Baseline Apple Silicon MPS path with compiled step and queued chunkwise prefill.
- Early PyTorch JIT + Metal integration, unified Metal‑optimized variants.
- Initial optimizer harness and foundational docs.

Commit digest (high‑level)
- 4289f40 feat(mps): Apple/MPS compiled backends + queued chunkwise for xLSTM
- 7971cbb Implement PyTorch JIT + Metal integration for xLSTM
- bfde744 Research and implement PyTorch Metal kernel integration
- d2560bd Add complete Metal kernel xLSTM implementation
- 4915551 Add unified Metal‑optimized xLSTM implementation
- e7fecd0 Initial commit

[0.2.0]: https://github.com/your-org/xlstm/releases/tag/v0.2.0
[0.1.0]: https://github.com/your-org/xlstm/releases/tag/v0.1.0
