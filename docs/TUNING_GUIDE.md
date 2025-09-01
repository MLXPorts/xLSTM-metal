Tuning Guide: xLSTM on Apple/MPS

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

Overview
- This guide explains the key tunables for running xLSTM (mLSTM + sLSTM) on Apple Silicon using our MPS‑native compiled backends.
- The defaults favor correctness and stability; the knobs help you trade prefill throughput vs decode speed.

Update (2025-08):
- “metal” denotes a compiled PyTorch step executing on MPS (not handwritten Metal).
- Prefill uses either the queued or Ray chunkwise schedulers (both GPU‑only); decode loops the compiled step.

Chunkwise Backends (prefill)
- queued_compiled_steps: CPU thread pool dispatching compiled step kernels per chunk/head‑band. Stable and predictable.
- ray_compiled_steps: Ray actors in local_mode (in‑process) coordinating compiled step kernels. Similar performance, easier async pipelining.
- native_compiled_autograd: Fully compiled native chunkwise; can hit Metal arg limits on long sequences — use with caution.

Key Tunables
- chunk_size (default 32):
  - Smaller (16–32): Lower per‑kernel arg pressure; smoother on MPS; smaller Python overhead with Ray/queued.
  - Larger (48–64): Fewer calls, potentially higher prefill throughput; watch for compiler stability.
- heads_per_band (default 4):
  - More heads per band increases per‑kernel work; fewer actors/threads; can reduce scheduling overhead.
  - Fewer heads per band increases concurrency (more actors/threads); helps saturate GPU.
- workers (queued backend):
  - # of CPU coordinator threads (not math). Too few can starve GPU; too many can oversubscribe.
  - Typical sweet spot: 4–8 on M3 Ultra.
- streams (optional):
  - Dedicated MPS streams for queued backend; many torch builds don’t expose streams yet. Keep 0 unless supported.
- step/sequence kernels:
- step=metal (compiled) and sequence=native_sequence__metal (compiled loop) are the defaults for MPS.

Fusion & Inner Tiling (design)
- Pseudo‑kernel fusion: we rely on `torch.compile` to fuse per‑timestep ops into a single step kernel.
- Inner tiling (`T_inner`): compile an unrolled block of steps (e.g., 4 or 8) to increase fusion windows; the driver loops blocks to cover the logical `chunk_size`.
- Canon mode: keep `chunk_size` fixed; maintain strict time order; use fp32 for (C,N,M) state. Runtime shrink is removed to preserve canonical behavior.
- Compile‑probe (planned): auto‑choose a safe `T_inner` at compile time if the Metal compiler signals argument/graph limits.
- See also: `docs/PYTORCH_MPS_FUSION_NOTES.md`.

Memory Watchdog
- Enabled by default; configure via envs. Soft threshold triggers `torch.mps.empty_cache()`; no runtime `chunk_size` changes are made.
- Hard threshold aborts to prevent unified memory exhaustion.
- Key envs: `XLSTM_MEM_SOFT_PCT`, `XLSTM_MEM_HARD_PCT`, `XLSTM_MEM_SOFT_MB`, `XLSTM_MEM_HARD_MB`, `XLSTM_MEM_POLL_MS`, `XLSTM_MEM_ACTION`.

CLI flags (local runner)
- `--mem-log`, `--mem-every`, `--mem-soft-pct`, `--mem-hard-pct`, `--mem-soft-mb`, `--mem-hard-mb`
- `--mem-action` to control soft actions (warn,empty_cache).

Effects on Inference
- Prefill throughput (tok/s) responds most to chunk_size, heads_per_band, and workers.
- Decode speed (tok/s) is less sensitive but can benefit from overhead reductions (fewer Python round‑trips, small sequence chunking).
- Always preserve per‑band sequential order to maintain exact numerics; exploit concurrency across bands and hierarchy.

Observability & Tooling
- Optimizer (scripts/optimize_mps.py) logs trials to runs/mps_opt/<run>/{run_meta.json, trials.jsonl, summary.csv, best.json}.
- Save continuations for each trial (scripts/save_outputs_for_trials.py) and judge quality (scripts/judge_outputs.py or scripts/judge_with_ollama.py).
- Runner stats (scripts/run_local_xlstm_mps.py --stats-log): per-step decode timing to CSV for plotting (step, dt_ms, cum_ms, inst_tok_s, avg_tok_s).
- Plots (scripts/plot_opt_results.py): max decode tok/s by chunk_size/heads/workers, prefill vs decode scatter.

Ray vs Queued
- Ray (`ray_compiled_steps`): default backend. Set `XLSTM_RAY_LOCAL_MODE=1` for in‑process execution; tune `heads_per_band`, `chunk_size`.
  - Cleanup: `XLSTM_RAY_AUTOSHUTDOWN=1` calls `ray.shutdown()` at the end of the run. If a crash leaves a local head running, `ray stop --force` cleans it up.
- Queued (`queued_compiled_steps`): legacy coordinator; tune `workers`, `heads_per_band`, `chunk_size`; optional `streams`.

Parameter Search Workflow
- Optimize: `scripts/optimize_mps.py --backend {ray|queued} --mode {ga|random}` writes results to `runs/mps_opt/<run>/`.
- Save continuations: `scripts/save_outputs_for_trials.py --run <run_dir> --outputs <run_dir>/outputs` regenerates text for each trial.
- Judge: `scripts/judge_outputs.py --outputs <run_dir>/outputs` computes quality/diversity metrics to `ratings.csv/jsonl`.
- Inspect/plot: `scripts/plot_opt_results.py` for throughput summaries and scatter plots.

Tips & Tricks
- Start conservative: chunk_size=32, heads_per_band=4, workers=6 (queued) or Ray with hpb=4.
- For long contexts (16k–32k): keep chunk_size <= 32 initially; tune heads_per_band & workers first.
- Use the optimizer to search; then save outputs and judge to balance speed & quality.
- Prefer local_mode Ray; avoids tensor copies; all math stays on GPU.
- Avoid large fused kernels: Metal arg limits can break compiled chunkwise — stick to many small kernels.

Env var mapping (Ray default)
- XLSTM_CHUNKWISE_BACKEND: selects chunkwise backend key
- XLSTM_MPS_WORKERS: CPU coordinator threads (queued backend)
- XLSTM_MPS_HEADS_PER_BAND: heads per task (parallelism granularity)
- XLSTM_MPS_STREAMS: MPS streams (optional)

Safety
- All compiled paths enforce GPU‑only execution. Experimental settings should be applied via explicit overrides.
