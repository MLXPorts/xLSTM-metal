# Ontology & Provenance Map

Purpose: show where each component comes from (ideas → invariants → code), how it is justified (formal/bio/engineering), and where the evidence lives (tests, journal, commits).

## North‑Star Invariants → Code
- Preserve→Gate: influence gates preserve values → CubeGatedBlock α‑blend; energy parity at wrapper boundary. See 22, 06.
- Budget→Brains: explicit ponder/energy budgets → ACTHaltingHead + PonderTrainer; energy audits. See 05, 06.
- Time Writes The Story: Z5 base‑5 envelope; boundary‑only commits → scheduler `boundary_commit_mask`. See 22, 23.
- Determinism: fixed schedules + trace → `trace_hash`, Research Journal. See 22, 24.

## Component Lineage
- HRMXLSTM (wrapper)
  - Origin: Lift invariants into xLSTM stack; place influence gate at stack boundary; expose halting/energy telemetry.
  - Formal hooks: quotient/carry→Z5; value‑preserving blend; halting analogue. See 22 §10–11.
  - Bio mapping: carry spike and comb windows. See 23.
  - Code: `src/lnn_hrm/xlstm_hrm.py`.
  - Tests/Evidence: wrapper demo; `test_act_energy_telemetry.py`, `test_wrapper_multiblock.py`.
  - Journal: 2025‑08‑29 entry.

- CubeGatedBlock + MemoryCube
  - Origin: content‑addressable residual predictor; write only at boundaries; gate by confidence.
  - Formal hooks: preserve→gate; remainder/phase → phase keys. See 22 §10.
  - Bio mapping: comb features as phase stencils. See 23.
  - Code: `src/lnn_hrm/{cube_gated_block.py,memory_cube.py}`.
  - Tests/Evidence: `test_scheduler_and_cube.py`, `test_memory_cube_behavior.py`.

- Scheduler (Z5)
  - Origin: base‑5 microcycle; “no 5 → carry” defines commit windows.
  - Formal hooks: quotient/carry mechanism. See 22 §10; mermaid sketch.
  - Code: `src/lnn_hrm/scheduler.py`.
  - Tests/Evidence: `test_scheduler_and_cube.py`.

- ACTHaltingHead + PonderTrainer
  - Origin: adaptive compute time for budgets; expose halting probs; add ponder term.
  - Formal hooks: halting and budget as thresholds. See 22 §11; 05.
  - Code: `src/lnn_hrm/{act_halting.py,training/ponder_trainer.py}`.
  - Tests/Evidence: `test_act_energy_telemetry.py`, `test_trace_and_logging.py`.

- Telemetry (energy/logger/trace)
  - Origin: auditable runs with energy monotone story and deterministic replays.
  - Formal hooks: energy monotonicity; determinism. See 06; 22 §11.
  - Code: `src/lnn_hrm/telemetry{.py,/logger.py,/trace.py}`.
  - Tests/Evidence: logging test; Research Journal.

- Preflight (MPS+Ray)
  - Origin: compiled xLSTM graphs have no native fallback; require Apple MPS and Ray.
  - Code: `src/lnn_hrm/preflight.py`; used in demos and smoke script.
  - Docs: 31_ci_device_matrix.md; CONTEXT_SEED.

## Where to Look (Derivation Sources)
- Formal: 22_fractal_diffusion_formalism.md (engineering mappings §10–12).
- Bio: 23_bio_rall_hcn_mapping.md (Rall unwinding; channel roles; base‑5 envelope).
- Stability/Safety: 06_stability_safety.md (invariants; UMA notes).
- ACT: 05_act_halting.md (head+policy; training notes).
- Design Rationales: 15_design_rationales.md (motivation snapshots).
- Journal: 24_research_journal.md (runs, telemetry, repro commands).
- CHANGELOG: dated changes/commits; complements the journal.

## Provenance Discipline (How to Extend)
When adding a component, include a short “Origin” note (in docs and module docstring):
- Invariants used (Preserve/Gate, Budget, Time) and why.
- Formal hooks (cite 22 sections) and bio hooks (cite 23) if applicable.
- Safety/telemetry expectations and the specific tests that demonstrate them.
- Pointer to a Journal entry or planned experiment.

Template (drop at end of a doc/module):
- Origin: …
- Invariants: …
- Formal/Bio: …
- Evidence: tests X, Y; journal YYYY‑MM‑DD.

