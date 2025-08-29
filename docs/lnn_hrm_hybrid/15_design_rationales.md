# Design Rationales and Trade-offs

This document explains key decisions and alternatives considered.

---

## 1. Residual prediction vs full output

- Residuals are better conditioned, centered near zero, and respect the block’s normalization; easier for cubes to learn incrementally.

Alternative: learn a full surrogate for the block; higher risk of drift, heavier capacity demands.

---

## 2. Gate inputs

- Include conf and novelty to reduce over-trust on out-of-distribution keys.
- LayerNorm on inputs stabilizes gate MLP.

---

## 3. Liquid scope

- Narrow, block-level liquid steps parallelize and avoid monolithic ODE coupling.

---

## 4. HRM schedule

- Reset L between cycles to force phase-structured computation and avoid premature fixed points.

---

## 5. UMA-aware cubes

- Keep the majority of entries on CPU; stage hot slices onto GPU/MPS.
- Eviction prioritizes low-confidence and stale entries to preserve coverage.

---

## 6. Audits vs overhead

- Audits are periodic and sampled to bound cost; heavy audits only on anomalies.

---

## 7. Persistence format

- NDJSON shards are simple and append-friendly; parquet is compact but needs batch writes.

