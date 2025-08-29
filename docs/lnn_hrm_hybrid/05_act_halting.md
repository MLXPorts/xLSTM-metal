# Adaptive Computation Time (ACT)

We implement a minimal ACT head for token‑level halting telemetry (and leave the door open to segment‑level policies). The current code projects token states to a halting probability and reports:

- act_prob_mean: mean σ(logit) over (B,L)
- act_open_rate: fraction above threshold

Training code can attach a ponder loss and, if desired, gate computation when act_mask is true. The halting head sits after the cube‑gated residual, so decisions see influence‑gated states.

---

## 1. Setup (token‑level head)

`ACTHaltingHead(d_model, threshold)` → (probs, mask, stats). Threshold defaults to 0.5.

We recommend wiring a ponder loss externally: L = L_task + λ·E[steps], keeping step budgets explicit in telemetry.

## 2. Segment‑level policy (optional Q‑head)

At segment m, state is z_m (H-state; optionally summary stats). Actions: {halt, continue}. Rewards: halt→1{correct}, continue→0 except at cap.

Q-head: Q̂_m = W_Q z_m + b (or a small MLP). Decision:

if m ≥ M_max → halt. Else if m ≥ M_min and Q̂_halt > Q̂_continue → halt; else continue.

M_min is ε-randomized to promote exploration; M_max is a hard cap.

---

## 3. Targets

Ĝ_halt = 1{ŷ_m = y};

Ĝ_continue = Q̂_{m+1, halt} if m ≥ M_max else max(Q̂_{m+1, halt}, Q̂_{m+1, continue}).

Loss per segment m: L_m = CE(ŷ_m, y) + λ_Q·BCE(Q̂_m, Ĝ_m)

Detach z between segments to implement the 1-step gradient approximation and avoid BPTT.

---

## 4. Stability

Bounded parameters via AdamW and post-norm RMSNorm; small learning rates for Q-head. No replay buffers or target nets used; empirical stability comes from segment-level short horizons.

---

## 5. Inference-time scaling

Increase M_max at inference for more “thinking”.
