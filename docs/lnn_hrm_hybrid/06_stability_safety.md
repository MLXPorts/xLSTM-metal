# Stability, Safety, and UMA-Aware Operation

This document enumerates invariants, numerical safeguards, and memory policies necessary for robust operation.

---

## 1. Numerical invariants

- Time constants τ strictly positive: τ = softplus(τ̂) + ε.
- Residual clamps: ||Δ||_∞ ≤ c (default c=1.0) for liquid updates.
- Gate rate limit: |α_t − α_{t−1}| ≤ ρ; hysteresis α_up > α_down.
- State dtype fp32; activations bf16; parameters bf16/fp16.

---

## 2. Canonical mode

- Fixed logical chunk sizes; strict time order; deterministic seeds; shrink disabled by default.
- Sub-chunking internal only; ensures reproducibility for audits.

---

## 3. UMA watchdog

- Soft threshold: warn and (optionally) shrink cube caches and KV budgets.
- Hard threshold: abort segment; clear caches; raise structured error.
- Telemetry: RSS, available, MPS alloc, cube bytes, shrink counts.

---

## 4. Memory cubes hygiene

- Max items per cube and overall budget; LRU eviction; TTL for stale entries.
- Backpressure propagation to gate (lower α_max) when budgets tighten.

---

## 5. Audits and rollbacks

- Shadow teacher recomputes sampled outputs to bound drift; if exceeded, demote α and purge offenders.
- Keep small holdout buffers within each cube for quick checks.

---

## 6. Security/Policy notes

- No mock components in production paths.
- Avoid “toy/simplified” phrasing in user-facing docs; describe exact behaviors.

---

## 7. Energy and Telemetry (Implementation)

- Energy budget: compute ||pre||² and ||post||² around the cube‑gated residual; ensure non‑increase on average. We log `energy_pre_gate` and `energy_post_gate` per pass.
- Gate audits: record `alpha_mean`, `conf_mean` per block; histogram per run.
- Determinism: record seeds, schedules (Z5 slots), and inputs; produce a trace hash.

## 8. Z5 Microcycle Discipline

- Five slots (0–4) with “no 5 → carry”: only allow memory‑cube commits on slot==4.
- Scheduler: `boundary_commit_mask(times)` marks boundary positions; `CubeGatedBlock.update` respects this mask.
