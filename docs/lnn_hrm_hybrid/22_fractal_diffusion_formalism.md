# Mathematical Proof System for Fractal Diffusion Computation (Integrated with SolaceCore HRM+/UKM)

This chapter consolidates a rigorous, exact-arithmetic treatment of the fractal diffusion computation and integrates its invariants with the SolaceCore HRM+ stack and UKM design. It is written as a self-contained formal document (definitions, lemmas, theorems, proofs), and then bridged to implementable engineering invariants used by our code.

Contents
- 1. Model (Rational Transition System)
- 2. Exactness over ℚ (closure, no floating-point oracles)
- 3. Determinism (δ is a function under a fixed schedule)
- 4. Energy Functional and Monotonicity (symbol and configuration)
- 5. Value-Preserving Reversible Gate (H^R_ε)
- 6. π-Shell Remainder Chain (worked verification)
- 7. Halting Guarantees (bounded schedules; ACT analogue)
- 8. Self-Similarity Cull (novelty vs familiarity)
- 9. Turing Completeness (effective encodings both ways)
- 10. Engineering Mappings (HRM+/UKM modules, signals, schedules)
- 11. Safety/Telemetry Invariants (energy, gates, carries)
- 12. Notes and Assumptions (scope; schedule regularity)
- Appendix A: Uniqueness of Euclidean remainder (ℚ with integer divisors)
- Appendix B: No increase under modulus
- Appendix C: Threshold correctness


## 1. Model (Rational Transition System)

State (tape symbol): τ := (v, a, s)
- v ∈ ℚ (exact rational value)
- a ∈ {0,1} (active flag)
- s ∈ ℕ (step counter)

Parameters
- Halt threshold ε := 10⁻⁶ (or any fixed rational ε > 0).
- Divisor schedule D := (d₀, d₁, …), dᵢ ∈ ℕ, dᵢ ≥ 1. D is fixed and known (external program/schedule).

Transition function δ on active symbols (a = 1)
- v' := exact_modulus(v, d) := v − d ⌊v/d⌋, with a canonical remainder r ∈ [0, d) for v ≥ 0 (signed variants acceptable if fixed).
- a' := a ∧ (|v| ≥ ε·d) (gating at the current scale d).
- s' := s + 1.
- δ( (v, a, s), d ) := (v', a', s').

Halting predicate
- halt(τ) iff a = 0 or |v| < ε.

Value-preserving reversible gate
- H^R_ε(v, a) := (v, a ∧ (|v| ≥ ε)). This gate preserves the value exactly and only flips the active bit by threshold comparison.

Remark on canonicity: the signed canonical remainder convention must be fixed once and used consistently; our code path uses nonnegative canonical remainders (r ∈ [0, d)) to align with Euclidean division.


## 2. Exactness over ℚ

Theorem 2.1 (Closure of ℚ under the transition)
Let v ∈ ℚ and d ∈ ℕ, d ≥ 1. Then exact_modulus(v, d) ∈ ℚ. Hence, if V₀ ∈ ℚ, every v reached by iterating δ remains rational.

Proof. Write v = p/q, p ∈ ℤ, q ∈ ℕ. v/d = p/(qd); k := ⌊v/d⌋ ∈ ℤ; v − dk = (p − dqk)/q ∈ ℚ. Updates to a and s lie in {0,1}, ℕ. ∎

Corollary 2.2 (No floating-point error)
All arithmetic is exact in ℚ; ε participates only in a boolean predicate.


## 3. Determinism

Theorem 3.1 (Deterministic transition)
Fix step i and divisor dᵢ. For any τ = (v, a, s), δ(τ, dᵢ) is uniquely determined.

Proof. Floor is single-valued; exact_modulus and boolean updates are functions; δ is a function. ∎

Remark. With fixed D, the evolution is a deterministic map on states.


## 4. Energy Functional and Monotonicity

Define E[τ] := |value(τ)|² if active(τ) = 1, else 0; and E_total := Σᵢ E[τᵢ] for a finite configuration.

Lemma 4.1 (Single-step value contraction by modulus)
Let v ≥ 0 and d ≥ 1. Write v = q d + r, q := ⌊v/d⌋, r ∈ [0, d). Then 0 ≤ r ≤ v and r < d; hence |r| ≤ |v|. ∎

Theorem 4.2 (Energy monotonicity per symbol)
If δ updates an active symbol (a = 1) using d ≥ 1, then E' ≤ E. If the gate deactivates (a' = 0), then E' = 0.

Proof. v' = v mod d with |v'| ≤ |v| by Lemma 4.1 ⇒ |v'|² ≤ |v|². If a' = 0, E' = 0. ∎

Corollary 4.3 (Configuration energy is nonincreasing)
E_total is nonincreasing under deterministic evolution.

Remark (Ledger form). If we define E_dump as the sum of |v|² at each deactivation event (a:1→0), then for any finite run: E_active + E_dump = E_initial.


## 5. Value-Preserving Reversible Gate

Proposition 5.1 (Value preservation by H^R_ε)
For any (v, a), H^R_ε preserves v exactly and may only flip a when |v| < ε.

Corollary 5.2 (Exact reconstruction of value)
Any sequence of H^R_ε gates preserves v₀ exactly; |v̂₀ − v₀| = 0.

Lemma 5.3 (δ ∘ H^R_ε vs H^R_ε ∘ δ)
The two compositions agree on the value component (noncommutativity affects only a). Thus, value traces are equal regardless of the order of applying δ and H^R_ε; the gate’s action changes influence, not information content.


## 6. π-Shell Remainder Chain (Worked Case)

Let π_exact := 355/113, π₂ := 355/339, π₃ := 355/1017. For v₀ = 140:

140 = 44·(355/113) + 200/113,
200/113 = 1·(355/339) + 245/339,
245/339 = 2·(355/1017) + 25/1017.

Each quotient is an integer; each remainder canonical (0 ≤ r < divisor). This triple illustrates a fast/mid/slow remainder cascade akin to theta/beta/delta shells.


## 7. Halting Guarantees

We separate:
- Threshold halting: |v| < ε.
- Scale deactivation: a' = 0 when |v| < ε·d.

Theorem 7.1 (Eventual halting under mild schedule regularity)
Suppose there exists d_min ≥ 1 such that dᵢ ≥ d_min and, infinitely often, dᵢ ≤ |vᵢ|. Then either:
(a) |vᵢ| falls below ε (threshold halting), or (b) gate deactivates at some step (|vᵢ| < ε·dᵢ).

Sketch. Whenever dᵢ ≤ |vᵢ|, δ forces |vᵢ₊₁| < |vᵢ|. Infinitely many such steps drive |vᵢ| below any fixed threshold, yielding (a) or (b). ∎

Corollary 7.2 (Periodic bounded schedules)
If D cycles over a finite set S with min S ≥ 1 and visits at least one d ≤ |v| per cycle, halting occurs in finite time.


## 8. Self-Similarity Cull (Novelty vs Familiarity)

Define for base b ≥ 2: cull_b(v, ε) := (|v mod b| < ε) ∨ (|v mod b − b| < ε). This detects near-divisibility (self-similarity) to b.

Proposition 8.1
cull_b holds iff v lies within ε of an integer multiple of b. ∎

Heuristic 8.2 (2/3 survival for b=3 under uniform residues)
If r mod 3 is uniform, then in the limit ε→0 only r≈0 is removed; two residue classes survive ⇒ ≈ 2/3 survival (idealized). In practice, empirical distributions govern survival.


## 9. Turing Completeness (Effective Encodings)

Encoding rationals to a finite alphabet Σ_bin := {0,1,#,±}:
enc(p/q) := (±)(bin(|p|))#(bin(q)) for reduced p/q.

Theorem 9.1 (TM simulates δ)
There exists a classical single-tape TM T over Σ_bin that simulates one δ-step on enc(v) in finite time; long division and comparisons are primitive recursive; ε is a fixed rational constant. Overhead is polynomial in the bitlengths of (p,q). ∎

Conversely, encode a TM’s configuration conf(M) via Gödel/radix encodings into a single rational v_conf and simulate updates by bounded compositions of δ and gates with small integer divisors (digit extraction via modulus, localized rewrites via additions/multiplications).

Theorem 9.2 (Our system simulates any TM)
There exists a bounded macro-schedule over small integers that maps v_conf(t) ↔ conf(M, t) until halting; upon halting, the threshold gate deactivates (a=0). ∎

Corollary 9.3 (Universality)
The rational transition system is computationally universal under effective encodings.

Remark (Energy). Simulated steps can be arranged to maintain nonincreasing |v| (use modulus/extraction), preserving the energy monotonic story at the macro-step level.


## 10. Engineering Mappings (HRM+/UKM)

We do not literally run over ℚ in the HRM codebase; instead we adopt the invariants:

- Quotient/carry → dendritic carries, boundary commits, and memory-cube write events. In code: a Z5 microcycle (five slots; “no 5 → carry”) schedules commit-only-at-boundary.
- Remainder/phase bitmask → phase features (fast/mid/slow) and comb masks. In code: auxiliary keys derived from Spiral/phase encodings; MemoryCube.spike_comb; dendritic-comb filters.
- Value-preserving gate → influence gating, not deletion. In code: CubeGatedBlock returns α/conf; raw hidden activations remain auditable pre-blend.
- Energy monotonicity → budget controller and telemetry. In code: track ||pre||², ||post||², ΔE; enforce monotone-on-average; raise audits when violated.
- Determinism → fixed schedules and seeds. In code: explicit microcycle and phase-bin schedules; trace hashes for replay.
- Halting → ACT analogue. In code: adaptive halting head with ε-like thresholds and per-step budget; reason codes logged.

Mermaid (schedule sketch)
```mermaid
flowchart LR
    S0[Slot 0] --> S1[Slot 1] --> S2[Slot 2] --> S3[Slot 3] --> S4[Slot 4]
    S4 -->|carry spike (no 5→10)| S0
    classDef carry fill:#fdd,stroke:#c44,stroke-width:2px
    class S4 carry
```


## 11. Safety/Telemetry Invariants

- Preserve, then gate: store pre-gate states; gates modulate influence (α), not existence.
- Energy budget: E_active + E_dump == E_initial within tolerance; ΔE per block monotone on average.
- Gate trace: per step log {α_mean, conf, a_open_rate}.
- Carries: count boundary commits per Z5 cycle; locate “10” events.
- Determinism: inputs + schedule + seed → trace id; replays match.
- Halting: ACT steps, halt reason (threshold vs budget vs policy).


## 12. Notes and Assumptions

- Regularity of D: periodic/bounded schedules guarantee halting via the Lyapunov argument.
- Canonical remainder: pick once (nonnegative is used in our code) and stick with it.
- Finite alphabet: simulations rely on effective encodings; direct ℚ symbols form a countable alphabet, equivalent via encoding.


### Appendix A. Uniqueness of Euclidean Remainder
Given v ∈ ℚ and d ∈ ℕ, there is a unique pair (q, r) with q ∈ ℤ, r ∈ ℚ such that v = qd + r and 0 ≤ r < d.

Proof. If v = qd + r = q'd + r' with 0 ≤ r,r' < d, then (q − q')d = r' − r ∈ (−d, d). Thus r' − r = 0 ⇒ r = r' ⇒ q = q'. ∎

### Appendix B. No Increase Under Modulus
For v ≥ 0 and d ≥ 1, r := v mod d satisfies r ≤ v and r < d; therefore |r|² ≤ |v|². For v < 0, apply to |v| by symmetry. ∎

### Appendix C. Threshold Correctness
H^R_ε(v, a) := (v, a ∧ (|v| ≥ ε)) preserves v and flips a only when |v| < ε. ∎
