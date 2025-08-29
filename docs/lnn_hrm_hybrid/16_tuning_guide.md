# Tuning Guide

Hyperparameters and heuristics for stable performance.

---

## 1. Cube parameters

- max_items: 16k–64k per block initially; scale with memory
- top‑K: 8–16
- similarity temperature τ_sim: 0.1–0.2
- write policy: write all during warmup; after, sample writes when conf < c_write

---

## 2. Gate

- α cap warmup from 0.0→0.2 over first W batches
- rate ρ = 0.05; hysteresis c_up=0.8, c_down=0.5

---

## 3. Liquid

- residual clamp: 1.0; try 0.5 if oscillations
- τ init: 1.0; allow per-dim learning

---

## 4. HRM schedule

- T inner steps: 2–8; N cycles: 1–4 for small models
- Segment count M_max: 2–8; with ACT on, train at smaller M_max then increase at inference

---

## 5. Optimizer

- AdamW, lr=1e-3 for small models; weight decay 0.01; post-norm everywhere

---

## 6. Diagnostics thresholds

- drift δ: choose percentile of teacher residuals on validation, e.g., 95th
- demote δ_α: 0.1; promote δ_α: 0.05

