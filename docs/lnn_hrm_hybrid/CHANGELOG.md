# CHANGELOG — HRM+/xLSTM

## 2025-08-28
- Add Z5 boundary-commit scheduler and wire into CubeGatedBlock (`allow_commit`).
- Add ACTHaltingHead and energy audits; return telemetry from wrappers.
- Add HRMXLSTM wrapper around official xLSTMBlockStack (boundary cube gating + ACT).
- Add formalism (22_) and bio mapping (23_) docs; update stability notes.

Commits: f729fff, 509c6b1

## 2025-08-29
- Verified MPS path: wrapper demo and tiny trainer run; telemetry logged to `runs/telem_demo`.
- Added research journal `24_research_journal.md` with exact repro commands and snapshots.
- Confirmed tests pass: `test_scheduler_and_cube.py` (2/2), `test_act_energy_telemetry.py` (1/1 on MPS; skips on CPU).
