Logging & Observability

Artifacts per optimizer run (runs/mps_opt/<run>/)
- run_meta.json: backend, bounds, repeats, seed, prompt, etc.
- trials.jsonl: one JSON per trial (params + metrics)
- summary.csv: tabular summary of trials.jsonl
- best.json: updated on improvements only (params + metrics)

Outputs & Judging
- Save continuations per trial: scripts/save_outputs_for_trials.py
  - outputs/<encoded-params>.txt files
- Internal judge (model‑as‑critic): scripts/judge_outputs.py
  - ratings.jsonl / ratings.csv with avg_logprob, ppl, distinct‑2/3
- External judge (Ollama): scripts/judge_with_ollama.py (e.g., qwen3-coder:30b)
  - ratings_ollama.jsonl / ratings_ollama.csv with coherence/relevance/fluency/overall

Runner stats (per‑step decode)
- Enable CSV logging: scripts/run_local_xlstm_mps.py --stats-log path.csv [--stats-every N]
  - Columns: step, dt_ms, cum_ms, inst_tok_s, avg_tok_s
  - Useful for MATLAB/Python plots (throughput stability, warm/cold phases)

Memory telemetry
- Global CSV: scripts/run_local_xlstm_mps.py --mem-log mem.csv [--mem-every 200]
  - Columns: ts,rss_mb,avail_mb,total_mb,mps_alloc_mb,mps_reserved_mb
  - Works with drivers’ watchdog that dynamically shrinks chunk_size and aborts on hard limit.
  - Control soft actions and shrink: `--mem-action warn` to disable cache clears, `--no-mem-shrink` to keep chunk size fixed, `--min-chunk` to bound shrink.

Ray Dashboard (optional)
- Start with: `--ray-dashboard [--ray-dashboard-port 8265] [--ray-keep-alive]`.
- URL: http://127.0.0.1:8265 shows live tasks/actors, logs, and custom metrics.
- Metrics exposed: `xlstm_tok_s_prefill`, `xlstm_tok_s_decode` (gauges), plus any Ray-native stats.
- Safety on MPS: dashboard implies `XLSTM_RAY_LOCAL_MODE=0`. Our driver auto-terminates actors if you keep the head alive (`--ray-keep-alive`) to avoid GPU memory pinning.

Plotting
- scripts/plot_opt_results.py --run runs/mps_opt/<run>
  - max_decode_by_chunk_size.png, max_decode_by_heads_per_band.png, prefill_vs_decode.png

Recommended Workflow
1) Optimize (GA/random/bandit), observe summary.csv
2) Save outputs per trial for quality inspection
3) Judge internally and/or via Ollama (Qwen judge)
4) Merge speed (summary.csv) and quality (ratings.csv) on filename slug
5) Pick Pareto‑optimal settings per context length; update defaults

Terminal TUI (xltop)
- Live monitor in a terminal: `python scripts/xltop.py` (curses UI).
- Fallback/no-curses: `python scripts/xltop.py --no-curses --poll 1.0` or single snapshot with `--once`.
- Controls: q quit, p pause, s set-interval, l toggle CSV log, C clear MPS cache, r ray status, k ray stop --force, K kill PID, h help.
