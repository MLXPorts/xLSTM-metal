# Long Pseudocode Reference

This appendix collects end-to-end pseudocode variants for training and inference in greater detail.

---

## 1. End-to-end training with cubes and liquid (single segment)

```
for batch in loader:
  e = f_I(x)
  h = e
  for b in blocks:
    # teacher mapping
    y_T = block_b(h)
    # cube
    k = phi_b(h)
    pred, conf = cube_b.query(k)
    # liquid
    y_L = liquid_b(h)
    # gate
    alpha = gate_b(h, pred, conf)
    # blend
    y = (1 - alpha) * y_T + alpha * (h + pred)
    # teacher residual for write
    delta = (y_T.detach() - h.detach())
    cube_b.update(k, delta)
    h = y
  y_hat = head(h)
  loss = CE(y_hat, y_true) + λ_cons * ||y - y_T||
  loss.backward(); opt.step(); opt.zero_grad()
```

---

## 2. HRM with segments and ACT

```
z = (z_H0, z_L0)
for m in 1..M_max:
  for n in 1..N:
    for t in 1..T:
      h, z_L = liquid_step(h, z_L)
    z_H = H_update(z_H, h)
  y_hat = head(z_H)
  L = CE(y_hat, y)
  if ACT:
     Q = Q_head(z_H)
     G = targets(Q, y_hat, y, m)
     L += λ_Q * BCE(Q, G)
  L.backward(); opt.step(); opt.zero_grad()
  if should_halt(Q, m): break
  z = detach(z)
```

---

## 3. Inference with audits

```
with no_grad():
  h = e
  for b in blocks:
    y_T = block_b(h)
    k = phi_b(h)
    pred, conf = cube_b.query(k)
    alpha = gate_b(h, pred, conf, read_only=True)
    y = (1 - alpha) * y_T + alpha * (h + pred)
    h = y
  y_hat = head(h)
  if audit_due():
     y_T_full = teacher_forward(e)
     drift = norm(y - y_T_full)
     if drift > δ:
        demote_alpha(); prune_cubes()
```

---

## 4. Cube maintenance and UMA handling

```
def maintenance():
  if uma_soft(): shrink_cubes(global_factor)
  if uma_hard(): clear_caches_and_abort()
  evict_low_utility_entries()
  persist_shards_if_configured()
```

---

## 5. Gate hysteresis update

```
if conf > c_up and recent_err < ε_up:
   alpha_max = min(1.0, alpha_max + δ)
elif conf < c_down or drift > ε_down:
   alpha_max = max(alpha_min, alpha_max - δ)
alpha = clamp(alpha_prev + clip(alpha_raw - alpha_prev, -ρ, +ρ), alpha_min, alpha_max)
```

