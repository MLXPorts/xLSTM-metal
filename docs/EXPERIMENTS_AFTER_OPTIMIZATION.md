Title: Post‑Optimization Experiments for Extended Context (sLSTM + Multi‑Kernel)

Purpose
- Capture safe, high‑impact experiment ideas to try after baseline MPS optimization is complete.
- Focus on designs that preserve recurrence coherence while exploiting fast, parallel summaries.

Ground Rules
- Keep core recurrence (C,N,M) strictly sequential per head‑band for exact numerics.
- Run bands and hierarchy levels concurrently (Ray/queued), but never feed out‑of‑order states into (C,N,M).
- All math on GPU (MPS); compiled kernels small and numerous (no large fused graphs).

Experiments

1) Side‑Channel Summaries (Recommended, Low Risk)
- Compute medium/long summaries in parallel (hierarchical prefill) and inject as auxiliary conditioning vectors into sLSTM gate preactivations (e.g., additive bias/adapters).
- Update side‑channel as fresher summaries arrive; do not replace (C,N,M).
- Optional: fine‑tune with stochastic delays/noise on the summary channel.

2) Dual‑Path Decode (Coarse + Exact)
- Maintain two paths: a fast coarse path for long‑range context and a strict sequential path for recent tokens.
- Blend logits via a small gate or fixed interpolation; strict path remains authoritative.
- Correct as strict path catches up (re‑ranking/interpolation).

3) Hierarchy as Context Tokens
- Treat medium/long summaries as compact “context tokens” prepended (or injected) during decode.
- Refresh opportunistically; strict sLSTM state still advances in order.

4) Multi‑Timescale Memory (Comb‑Filter Analogy)
- Maintain working memory pools with different retention rates: recent, relevant, compressed, persistent.
- Expose a lightweight mixer that combines low‑frequency (compressed) and high‑frequency (recent) signals.
- Implement via the pool manager without violating per‑band recurrence.

5) Dendritic (Tree) Ordering via Transition Operators (Research Track)
- Define a chunk‑transition operator T mapping (C,N,M)→(C’,N’,M’) that is numerically stable and approximately associative.
- Pipeline: Pass 1 compute T for all chunks in parallel; Pass 2 prefix‑scan T to propagate states; Pass 3 (optional) recompute outputs with correct carries.
- Requires careful math and likely training to be robust; keep as a controlled prototype.

Scheduling Notes
- Ray driver must chain chunks sequentially per band (one active chunk per band) and keep many bands in flight (small global in‑flight window).
- Queued backend already respects sequential per‑band constraints; tune workers, heads_per_band, and chunk_size.

Avoid (Until Retrained)
- Replacing the core (C,N,M) with speculative/out‑of‑order states — breaks causality and drifts without training.
- Large fused/long compiled sequence kernels — re‑introduces Metal argument limit/codegen failures.

Measurement
- For each experiment, track: prefill tok/s, decode tok/s, perplexity vs length, retrieval Recall@K, numerical drift in states, stability.
- Keep parity scripts for sLSTM/mLSTM handy; roll back any experiment that harms coherence or throughput.

