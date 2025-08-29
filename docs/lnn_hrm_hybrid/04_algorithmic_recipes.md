# Algorithmic Recipes: Training, Inference, Gate Schedules, and Audits

This document turns the formalism into step-by-step procedures with practical hyperparameters.

---

## 1. Training (supervised, no ACT)

Inputs: dataset D of (x,y). Initialize θ_T, θ_LNN, cubes C^b empty.

For each batch:
1) Encode inputs e = f_I(x)
2) For each Transformer block b:
   - Compute h_in^b → y_T^b (teacher)
   - Compute k^b = φ(h_in^b)
   - Query cube C^b: (Δŷ^b, conf^b)
   - Liquid step: y_L^b from LNN
   - Gate α^b ← Γ(h_in^b, Δŷ^b, conf^b)
   - Blend y^b ← (1−α^b)·y_T^b + α^b·(h_in^b + Δŷ^b)
   - Online update: write (k^b, Δy^b=y_T^b−h_in^b)
3) Compute output ŷ and loss CE(ŷ,y) + optional consistency λ||y − y_T||
4) Backprop; optimizer step

Hyperparams:
- α cap warmup: cap α ≤ 0.2 for first W steps, then linearly to 0.8.
- Cube size per block: 16–64k entries; top‑K=8–16; τ_sim=0.07–0.2.
- Residual clamp: ||Δ||_∞ ≤ 1.0

---

## 2. Training (HRM with segments; optional ACT)

We train in segments m=1..M. Between segments, detach z.

Loop over segments:
1) Run N cycles × T inner steps with cubes+liquid enabled.
2) Decode ŷ_m; compute CE loss.
3) If ACT on: compute Q̂_m and BCE to targets Ĝ_m.
4) Sum losses; backprop; step.
5) Detach (z_H,z_L) before next segment.

---

## 3. Inference

- For each sample, run with cubes in read mode (no writes), α bounded by latest trust.
- Optionally enable periodic audits (shadow teacher recompute) per K tokens.
- Increase M_max for more segments to improve hard instances.

---

## 4. Gate schedule and hysteresis

Let α_raw from MLP; apply:
- α_t = clamp(α_{t−1} + clip(α_raw − α_{t−1}, −ρ, +ρ), α_min, α_max)
- Promotion if conf > c_up and recent_err < ε_up → α_max ← α_max + δ (≤ 1.0)
- Demotion if conf < c_down or drift > ε_down → α_max ← α_max − δ

Typical values:
- ρ = 0.05 per step; c_up=0.8; c_down=0.5; ε_up=low; ε_down=medium

---

## 5. Cube maintenance

- On each step: enqueue writes; flush at end of batch.
- Evict: LRU of low-confidence/low-usage entries; preserve diverse keys (avoid collapse).
- UMA pressure: watchdog signals soft/hard thresholds → shrink cubes globally.

---

## 6. Audits

- Every A tokens or seconds, sample S positions, recompute y_T; measure drift = ||y − y_T||.
- If drift > δ for P consecutive audits: halve α_max and purge top contributors.

---

## 7. Minimal pseudocode

```python
def cube_gated_block(h_in, block, cube, liquid, gate_mlp, train: bool, teacher_consistency=0.0):
    # Teacher path
    y_T = block(h_in)
    # Keys and cube pred
    k = key_proj(h_in).view(-1, d_key)
    pred, conf = cube.query(k)
    pred = pred.view_as(h_in)
    # Liquid step (optional)
    y_L = liquid_step(h_in)
    # Gate
    feats = torch.cat([LN(h_in), LN(pred), conf.view(*h_in.shape[:-1],1)], dim=-1)
    alpha = torch.sigmoid(gate_mlp(feats)).clamp(0, 1)
    # Blend (residual form)
    y = (1 - alpha) * y_T + alpha * (h_in + pred)
    # Consistency loss
    loss_cons = teacher_consistency * (y - y_T).pow(2).mean()
    if train:
        delta_y = (y_T.detach() - h_in.detach()).view(-1, d_val)
        cube.update(k, delta_y)
    return y, loss_cons, alpha.mean().item(), conf.mean().item()
```

---

## 8. Debug & telemetry

- α_mean, conf_mean, cube_size, write_rate, evict_rate, drift.
- PR for z_H/z_L; forward residual across steps.

