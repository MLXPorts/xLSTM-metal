Title: sLSTM + Multi‑Kernel Scheduling for Extended Context on Apple/MPS

Objectives
- Extend effective context for xLSTM (mLSTM + sLSTM) to 20K–100K+ tokens while maintaining coherence and throughput on Apple Silicon (MPS).
- Leverage sLSTM’s exponential gating and state‑tracking advantages with a multi‑kernel scheduling strategy that uses many small compiled kernels (no monolithic fused kernels).
- Keep all math on GPU (strict GPU‑only); measure prefill/decode throughput and retrieval quality.

Baseline & Metrics
- Establish current performance (2K–16K) with Ray and queued backends:
  - Throughput: prefill tok/s, decode tok/s
  - Footprint: peak memory, per‑token state footprint
  - Quality: perplexity vs length, retrieval/recall on associative tasks, numerical drift
- Scripts: `scripts/bench_mps.py`, `scripts/bench_mps_ray.py`, `scripts/optimize_mps.py`

Phase 0 — Groundwork
- sLSTM compiled foundation (done): strict GPU guards, parity tests.
- Baselines: run sweeps on M3 Ultra for 2K–16K contexts to lock in reference metrics.
- Add prompt length sweeps to simulate long prefill (e.g., 8K/16K/32K/64K) and capture stability.

Phase 1 — Hierarchical Prefill (chunk → summary → compression)
- Short‑term (fine):
  - Process 32–64‑token chunks with compiled step kernels (mLSTM/sLSTM); emit per‑chunk states (y,c,n,m).
  - Schedule with Ray actors (local_mode=1) or queued backend; keep kernels tiny, concurrent.
- Medium‑term (summarization):
  - Group N chunks (e.g., 8) and compute a single representative “medium state” via a small compiled sLSTM summarizer over concatenated/pooled features.
- Long‑term (compression):
  - Fold M medium states (e.g., 8) into one “long state” via sequential or tree‑scan reduction.
- Expose a `HierarchicalSLSTMScheduler` that returns: H_recent (dense), M_medium (summaries), G_long (global).

Phase 2 — Working Memory Pools (recent / relevant / compressed / persistent)
- Memory pools:
  - recent: dense buffer of last ~1–2K tokens
  - relevant: salience‑selected tokens/states
  - compressed: older context summaries (medium/long)
  - persistent: entities/facts (long‑lived, slow updates)
- Parallel updates per chunk via small compiled kernels scheduled concurrently:
  - Update recent frequently, assess relevance at medium frequency, compress old at low frequency.
- Use sLSTM’s exponential gates to revise/retain and mix states across pools safely.

Phase 3 — Dynamic Context Allocation (salience‑driven budgets)
- Compute importance per chunk (norm deltas, heuristic keywords, lightweight scorers).
- Allocate budgets:
  - High importance → smaller chunk_size, more parallel kernels, higher retention (recent/relevant).
  - Low importance → larger chunk_size, more compression, lower retention and shorter retention horizon.
- Implement `DynamicContextManager` with a simple policy; later tune with GA/random search.

Phase 4 — API, Runners, Config
- API:
  - `extended_prefill(input, config) -> (H_recent, M_medium, G_long, updated_pools)`
  - `extended_step(token, pools) -> (logits, updated_pools)`
- Runners: flags to enable hierarchical prefill and memory pools; budgets and ratios adjustable from CLI.
- Backends: both Ray (`chunkwise--ray_compiled_steps`) and queued (`chunkwise--queued_compiled_steps`) are supported.

Phase 5 — Validation & Scaling
- Tasks:
  - Multi‑Query Associative Recall: scale >256 KVPs (target 1K–4K) with hierarchical prefill and pools.
  - Sequence extrapolation: 2K → 16K → 32K → 64K → 100K contexts (synthetic + real corpora).
- Metrics:
  - Retrieval: Recall@K, accuracy vs distance
  - Language quality: PPL vs length, qualitative coherence
  - Performance: prefill/decode tok/s, memory use, kernel stability (no Metal arg limit issues)

Engineering Plan
- Code layout:
  - `xlstm_official_full/extended_context/`
    - `hierarchical.py`: orchestrator (Ray actor or queued driver) using compiled step kernels
    - `summaries.py`: compiled sLSTM/mLSTM summarizers for medium/long reductions
    - `memory_pools.py`: pool management, policies, retention/eviction
    - `drivers/`: Ray actors (`ray_actors.py`), queued (`queued.py`), utilities
- Kernel design:
  - Prefer compiled step kernels + small summarizers; avoid large fused graphs.
  - Optional argument packing per compiled wrapper to minimize Metal buffer arguments.
- Scheduling strategy:
  - Ray actors (local_mode=1) preferred; CPU threads for queued remain available.
  - Keep chunk sizes small (32–64) and parallelism tuned via optimizer.

Parameter Optimization & Long‑Context Sweeps
- Use `scripts/optimize_mps.py` (GA + random) to maximize decode tok/s.
  - Ray: tune `heads_per_band`, `chunk_size`
  - Queued: tune `workers`, `heads_per_band`, `chunk_size`
- Extend sweeps to longer contexts:
  - Prefill prompt lengths: 8K, 16K, 32K, 64K (repeat tokens or synthetic streams)
  - Objectives: blend throughput and quality (e.g., `score = decode_tok_s * w1 + recall@K * w2`)
- Record best configs per context scale; persist into docs and runner defaults.

Milestones (PRs)
1) Hierarchical Prefill MVP
   - Implement orchestrator + medium/long summarizers
   - Integrate flags and timing output; validate stability to 32K contexts
2) Memory Pools + Dynamic Context
   - Add pool manager and salience scoring + policy allocator
   - Validate associative recall at larger KVP scales (≥1K)
3) Quality + Tuning
   - Add argument packing option; run GA/random sweeps across parameters and context lengths
   - Document best defaults; finalize Apple guide with recommended settings

Risks & Mitigations
- MPS compile instability for large kernels → use many small compiled kernels, argument packing, and Ray/queued concurrency.
- Numerical drift over very long contexts → use sLSTM’s stable gating/normalization; monitor state deltas.
- Coordination overhead → prefer local_mode Ray actors and keep tensors in‑process; keep kernels small and streams optional.

Current Tools
- Backends: `chunkwise--ray_compiled_steps`, `chunkwise--queued_compiled_steps`, `native_sequence__metal`, `metal` (step)
- Runners: `scripts/run_local_xlstm_mps.py`, `scripts/xlstm_run.py`
- Bench/Opt: `scripts/bench_mps.py`, `scripts/bench_mps_ray.py`, `scripts/optimize_mps.py`

