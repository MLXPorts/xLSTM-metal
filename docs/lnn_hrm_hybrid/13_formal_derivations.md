# Formal Derivations and Proof Sketches

This appendix gathers detailed derivations, assumptions, and proof sketches supporting stability, gating behavior, and segment-level gradient approximations.

---

## 1. Liquid dynamics discretization

Starting from

dx/dt = -[(1/τ) + f(x,I,t;θ)] ⊙ x + f(x,I,t;θ) ⊙ A

We consider an explicit-implicit mixed discretization over Δt:

Numerator: x_t + Δt · f(x_t,I_t;θ) ⊙ A

Denominator: 1 + Δt · [(1/τ) + f(x_t,I_t;θ)]

Thus

x_{t+Δt} = [x_t + Δt f_t ⊙ A] / [1 + Δt (1/τ + f_t)]

Stability: With τ_i > 0 and bounded f, the denominator remains ≥ 1, ensuring contraction for small Δt. In practice, we avoid explicit Δt by using time-coded gates σ(−f t) blending g,h transforms.

---

## 2. HRM 1-step gradient approximation

Let z_* be fixed points of the L-then-H update chain. Denote F the implicit H-level mapping. IFT yields

∂z_* / ∂θ = (I − J_F)^{-1} ∂F/∂θ

Approximate with 1-step gradient by replacing (I − J_F)^{-1} ≈ I. This corresponds to retaining only the first term of the Neumann series. We then backprop only through the last L and H updates per segment, detaching earlier steps. Memory is O(1).

---

## 3. Gate boundedness

Let α_raw ∈ (0,1). Define rate-limited α_t via

α_t = clamp(α_{t−1} + clip(α_raw − α_{t−1}, −ρ, +ρ), α_min, α_max).

Then |α_t − α_{t−1}| ≤ ρ, so cumulative deviation after s steps ≤ sρ. With residual clamp on y_C contributions, total blended deviation from teacher remains bounded by O(sρc).

---

## 4. Drift audits

Define drift d = ||y − y_T||. If d > δ across P audits, then demote α_max ← max(α_min, α_max − δ_α) and purge top-contributing entries from cubes (as measured by ∂y/∂v approximations or similarity weights). This forms a feedback system stabilizing around acceptable drift.

---

## 5. Cube capacity and generalization

Let K be the number of entries, with normalized keys. Retrieval error under cosine top‑K selection relates to covering number of the key manifold. With LRU + diversity-aware eviction (e.g., min-sim threshold), we maintain coverage; confidence estimates approximate local density.

---

## 6. UMA budget and safety

Let B_total be byte budget for cubes. With per-cube caps B_b and a global cap, perform proportional reduction B_b' = B_b · (B_target / B_total) upon soft watchdog signals. Ensure no unbounded growth.

---

## 7. Participation Ratio (PR)

Given covariance eigenvalues {λ_i}, PR = (Σ λ_i)^2 / Σ λ_i^2. In our HRM, PR_H increases with task diversity while PR_L remains relatively stable, indicating hierarchical dimensionality separation.

---

## 8. Extended notation table

- d, d_k, d_v: dims of block, key, value
- K: number of cube entries
- top‑K: retrieval neighbors per query
- ρ: max α rate
- δ: drift threshold
- τ: time constants (liquid)
- Δy: residual target (teacher − input)

---

## 9. Limiting cases

- α→0: pure Transformer teacher.
- α→1 with perfect cube: block becomes a learned associative mapping.
- No cubes: pure liquid refinement of block output.

---

## 10. Convergence heuristics

- Under stationary input distributions, online ridge heads on cubes converge to least squares solutions over local neighborhoods, improving residual predictions; hysteresis prevents oscillations in α.

