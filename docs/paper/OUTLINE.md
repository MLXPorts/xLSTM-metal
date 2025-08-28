Title: Multi‑Kernel Scheduling for Extended‑Context xLSTM on Apple/MPS

Authors: <Your Names Here>

Abstract
- One paragraph summary of the approach: replacing monolithic kernels with many small compiled kernels; MPS‑native; strict recurrence; Ray/queued scheduling; extended context via hierarchy; improved throughput and stability.

1. Introduction
- Motivation: long‑context inference, RNN strengths vs Transformers, Apple/MPS constraints.
- Contributions (bulleted):
  - MPS‑native compiled backends for xLSTM (mLSTM+sLSTM), no Triton/Metal shaders.
  - Multi‑kernel scheduling (Ray/queued) to saturate GPU while preserving recurrence.
  - Optimizer with GA/random/bandit + full observability.
  - Extended‑context roadmap (hierarchical prefill, memory pools) and initial results.

2. Background & Related Work
- xLSTM, sLSTM gating/stability, long‑context RNNs.
- MPS/Inductor compilation constraints.
- Scheduling on GPUs (concurrent kernels, task parallelism).
- Energy‑based/boltzmann exploration (optional footnote).

3. Methods
3.1 MPS‑Native Compiled Backends
- mLSTM step/sequence, sLSTM compiled forward/step; strict GPU‑only guards.

3.2 Multi‑Kernel Scheduling
- Queued driver (threads), Ray actor driver (local_mode), per‑band sequential constraint, cross‑band concurrency, in‑flight window.

3.3 Parameter Optimization
- GA/random/bandit/boktzmann; observability; reward definition (decode tok/s, prefill tok/s).

3.4 Extended‑Context Strategy (Roadmap)
- Hierarchical prefill (chunk→medium→long), working memory pools, dynamic context allocation.

4. Experiments
4.1 Throughput Benchmarks (2K→16K→32K)
- Setups, prompts, hardware, flags, search budgets.

4.2 Ablations
- Ray vs queued, heads_per_band/workers/chunk_size sensitivities.
- (Optional) tiny sequence chunk, arg packing toggles.

4.3 Quality Checks
- MQAR (256→1K+ KVPs), perplexity vs length, coherence examples.

5. Results
- Tables: best configs, tok/s; plots: throughput vs prompt length; stability.

6. Discussion & Limitations
- Recurrence constraints; MPS compiler limits; generality; retraining needs for speculative paths.

7. Conclusion
- Summary and future work (transition operators, scan/monoid approach; CfC controller; energy‑guided scheduling).

Appendix
- Implementation notes, environment, reproducibility instructions.

