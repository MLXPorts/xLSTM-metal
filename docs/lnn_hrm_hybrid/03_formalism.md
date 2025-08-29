# Unified Formalism: HRM × Transformer × Liquid × Memory Cubes

This document formalizes the hybrid as a composition of modules with clear timing, interfaces, and guarantees. We blend HRM’s two-timescale loop with block-local liquid dynamics and associative cubes.

---

## 1. Objects and spaces

- Inputs: x ∈ X (sequences)
- Embeddings: e = f_I(x; θ_I)
- Block boundary states: h ∈ R^{d}
- HRM states: z_L ∈ R^{d_L}, z_H ∈ R^{d_H}
- Cube per block b: C^b = (K^b, V^b, S^b)
- Liquid parameters per block: θ_LNN^b; Transformer params θ_T

---

## 2. HRM timing

Let T be inner steps per cycle; N cycles per segment; M segments until halt.

For i = 1..(N·T):

- If i mod T ≠ 0: update L only (fast loop)
- If i mod T = 0: update L and then H once (slow step)

Segments repeat with deep supervision; between segments z is detached.

---

## 3. Block with cube and liquid

Given input h_in at a block:

1) Transformer teacher mapping: y_T = F_T(h_in; θ_T)
2) Key projection: k = φ(h_in)
3) Cube prediction: (Δŷ, conf) = C.query(k)
4) Liquid update: (z_L, y_L) = LNN(h_in, z_L; θ_LNN, t)
5) Blending gate α = Γ(h_in, Δŷ, conf, stats)
6) Output: y = (1−α)·y_T + α·(h_in + Δŷ)  or  y = (1−α)·y_T + α·y_L

Teacher residual target: Δy = y_T − h_in

Online update: C.update(k, Δy)

---

## 4. Gate definition

Let features r = [LN(h_in), LN(Δŷ), conf, novelty], where novelty = 1 − max_sim.

α = σ(w_2 · σ(w_1 · r + b_1) + b_2), with rate constraint |α_t − α_{t−1}| ≤ ρ and hysteresis thresholds α_up > α_down.

---

## 5. Assumptions

- Bounded weights and τ_i > 0 imply stable liquid updates.
- Cube predictions use bounded residual norms (train targets are teacher residuals, which are themselves bounded by architectural normalization).
- UMA watchdog enforces memory bounds; eviction maintains invariants.

---

## 6. Losses and training

- Deep supervision over segments: L = mean_m CE(ŷ_m, y) + λ_Q·BCE(Q̂_m, Ĝ_m) if ACT enabled.
- Optional consistency regularizer: ||y − y_T|| on early phases to prevent premature takeover.
- Cube write policy during training controlled by schedule (e.g., only when confidence < c* to populate diverse entries).

---

## 7. Halting (ACT)

Define a Q-head over z_H: Q̂(halt), Q̂(continue). At each segment m, halt if Q̂(halt) > Q̂(continue) and m ≥ M_min, or m = M_max. Targets defined by correctness and bootstrap.

---

## 8. Guarantees (informal)

- If α is bounded by hysteresis and audits, deviations from teacher remain bounded by residual caps; takeover cannot drift arbitrarily fast.
- With τ positive and residual clamps, liquid dynamics cannot diverge within a segment.

---

## 9. Inference-time scaling

- Increase M_max to spend more segments; cubes benefit (more reads), liquids get longer inner schedules (more refinement), while H integrates across cycles.

---

## 10. Notes on universality

Combining recurrence (liquid) with adaptive halting and hierarchical control yields unbounded effective depth given time; cubes provide a memory substrate to amortize computation across encounters.

