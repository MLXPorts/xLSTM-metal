# Context Seed — HRM+/xLSTM First-Date Primer

This one-pager is the minimal state needed to resume work after a context reset.

- North Stars
  - Preserve → Gate (value is sacred; influence is gated)
  - Budget → Brains (energy/ponder budgets before cleverness)
  - Time Writes the Story (Z5 microcycle; boundary-only commits)

- Core Modules (paths)
  - `src/lnn_hrm/xlstm_hrm.py` — HRM wrapper over `xLSTMBlockStack` (cube gating + ACT + Z5 + telemetry)
  - `src/lnn_hrm/cube_gated_block.py` — MemoryCube blend; respects `allow_commit` mask
  - `src/lnn_hrm/memory_cube.py` — cosine top‑K store for residuals
  - `src/lnn_hrm/scheduler.py` — `boundary_commit_mask(times)` (slot==4)
  - `src/lnn_hrm/act_halting.py` — token‑level halting head (telemetry)
  - `src/lnn_hrm/telemetry.py` — `energy()` helper

- Demos
  - `python examples/xlstm_hrm_wrapper_demo.py` → y shape + telemetry dict

- Telemetry (returned by wrappers)
  - `alpha_mean`, `conf_mean` — cube gate
  - `act_prob_mean`, `act_open_rate` — ACT head
  - `energy_pre_gate`, `energy_post_gate` — energy audit (||·||²)

- Z5 Scheduler
  - `z5_slots(times)`, `boundary_commit_mask(times)`
  - Only update memory when `slot==4` (carry at rollover; “no 5 → 10”)

- Docs Index
  - 22_fractal_diffusion_formalism.md — exact arithmetic formalism + engineering mapping
  - 23_bio_rall_hcn_mapping.md — Rall/HCN mapping → invariants
  - 05_act_halting.md — token‑level ACT head + segment policy
  - 06_stability_safety.md — energy & Z5 discipline notes

- Quick Rebuild Steps
  1) Load `HRMXLSTM` with an xLSTM config; run one batch; confirm telemetry prints
  2) For transformer demo, use `examples/transformer_lnn_example.py` (optional)

- TODO (next stitches)
  - Add ponder loss hook and trainer aggregation of telemetry
  - Optional per‑block cube gating inside block stack, behind a flag
