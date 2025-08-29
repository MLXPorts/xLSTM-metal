# Edge Cases and Failure Modes

---

## 1. Cold start cubes

- Behavior: conf ≈ 0; α should stay near 0; predictions near zero residuals.
- Mitigation: warmup phase with writes; cap α small.

---

## 2. Key collapse

- Behavior: many keys cluster; overconfident but wrong predictions.
- Mitigation: diversity-aware eviction; add small noise to keys; temperature scaling.

---

## 3. Noisy teachers

- Behavior: residual targets high variance; cubes memorize noise.
- Mitigation: teacher smoothing/EMA; write only when teacher-confidence high (if available).

---

## 4. UMA pressure spikes

- Behavior: OOM risk.
- Mitigation: watchdog-triggered shrink; stop writes; demote α.

---

## 5. Liquid oscillations

- Behavior: unstable updates.
- Mitigation: reduce residual clamp; increase τ; add extra norm after step.

---

## 6. Gate flapping

- Behavior: α oscillates between extremes.
- Mitigation: rate limits; hysteresis; smoothing.

