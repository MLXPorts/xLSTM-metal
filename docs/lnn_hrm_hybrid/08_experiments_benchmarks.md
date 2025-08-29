# Experiments and Benchmarks

We outline synthetic and canonical tasks for sanity and capability checks, with recommended metrics.

---

## 1. Synthetic signals (sines + shifts)

- Generate multi-frequency sine mixtures with phase noise; predict next-step or transformed series.
- Use this to validate liquid stability, cube learning (residuals), and gate behavior.

Metrics: MSE, α_mean vs conf, cube hit rate, residual norms.

---

## 2. Sudoku (Extreme)

- Train with band/digit permutations; measure exact solution rate.
- Use tdoku difficulty buckets; study inference-time scaling with segments.

---

## 3. Maze (30×30)

- Optimal pathfinding tasks; verify shortest-path correctness; measure accuracy vs M_max.

---

## 4. ARC-like tasks

- Data augmentation via rotations/flips/color perms; two-guess protocol.
- Use cubes to cache recurring micro-pattern transformations across tasks.

---

## 5. Probes and diagnostics

- Forward residual profiles across steps; PCA trajectories.
- Participation Ratio (PR) for z_H vs z_L dimensionality.

---

## 6. Ablations

- No cubes vs cubes; no liquid vs liquid; both vs each.
- α schedules: fixed caps, hysteresis on/off.
- Cube sizes and top‑K variations.

