# HRM+/xLSTM Research Journal

This living journal captures test runs, telemetry snapshots, and small design notes. It complements CHANGELOG (what changed) with evidence (what worked) and exact repro steps.

## 2025-08-29 — Rehydrate + Smoke Tests (MPS)

- Host: macOS 15.7 (arm64)
- Python: 3.12.2 (conda base)
- PyTorch: 2.7.0
- CUDA: unavailable; MPS: available

### 1) Wrapper Smoke
- Command: `PYTHONPATH=. python examples/xlstm_hrm_wrapper_demo.py`
- Output shape: `y = (2, 16, 32)`
- Telemetry (sample):
  - `alpha_mean ≈ 0.5130`
  - `conf_mean ≈ 0.0000` (empty cube on first call)
  - `act_prob_mean ≈ 0.5071`
  - `act_open_rate ≈ 0.53125`
  - `energy_pre_gate == energy_post_gate ≈ 31.9999`

Notes:
- `boundary_commit_mask(times)` applied; first call leaves cube empty; conf≈0 is expected.

### 2) Tiny Trainer + Ponder
- Command: `PYTHONPATH=. python examples/train_with_ponder_demo.py`
- Steps: 5 on MPS
- Sample metrics (stdout):
  - `step 00 loss=4.2740 ce=4.2686 ponder=0.5353`
  - `step 01 loss=4.3020 ce=4.2974 ponder=0.4573`
  - `step 02 loss=4.4222 ce=4.4175 ponder=0.4726`
  - `step 03 loss=4.3210 ce=4.3157 ponder=0.5293`
  - `step 04 loss=4.3203 ce=4.3159 ponder=0.4366`

Artifacts:
- CSV/JSONL at `runs/telem_demo/{demo.csv,demo.jsonl}` with fields:
  - `alpha_mean, conf_mean, act_prob_mean, act_open_rate, energy_pre_gate, energy_post_gate, loss, ce, ponder, trace_hash, step, ts`

Example JSONL tail:
```
{"alpha_mean": 0.5252, "conf_mean": 0.3080, "act_prob_mean": 0.4726, "act_open_rate": 0.3438, "loss": 4.4222, "ce": 4.4175, "ponder": 0.4726, "trace_hash": "16e88f4c...f93", "step": 2}
{"alpha_mean": 0.5301, "conf_mean": 0.3672, "act_prob_mean": 0.5293, "act_open_rate": 0.5313, "loss": 4.3210, "ce": 4.3157, "ponder": 0.5293, "trace_hash": "5580439b...f4de", "step": 3}
{"alpha_mean": 0.5385, "conf_mean": 0.4301, "act_prob_mean": 0.4366, "act_open_rate": 0.1563, "loss": 4.3203, "ce": 4.3159, "ponder": 0.4366, "trace_hash": "32201971...2edb", "step": 4}
```

### 3) Targeted Tests
- Command: `PYTHONPATH=. pytest -q tests/test_scheduler_and_cube.py`
  - Result: 2 passed
- Command: `PYTHONPATH=. pytest -q tests/test_act_energy_telemetry.py`
  - Result: 1 passed (MPS path); will `skip` on pure CPU hosts

### Repro Notes
- Ensure `conda` base on PATH. For zsh users, mirror the PATH block into `~/.zshrc`.
- Always set `PYTHONPATH=.` when running examples/tests directly to resolve intra-repo imports.

### Short Design Echo
- Boundary-only commits: conf starts low; rises as slots accumulate.
- Energy audit parity at the wrapper boundary is expected (value-preserving gate).
- Ponder attaches cleanly via ACT telemetry; integration with a real head is next.

---

## 2025-08-29 — Serotonin (5‑HT) Gain Sweep (MPS)

- Script: `python examples/serotonin_sweep_demo.py` (levels 0.0, 0.5, 1.0; 8 steps each)
- Logs: `runs/5ht_sweep/level_{00,50,100}/run.{csv,jsonl}`
- Aggregation: `tools.telem.aggregate` → `docs/lnn_hrm_hybrid/_telem_5ht/{level}/`

Findings (means over steps)
- 5‑HT 0.0 → α_mean≈0.525; open_rate≈0.551; energy_post≈31.91
- 5‑HT 0.5 → α_mean≈0.360; open_rate≈0.395; energy_post≈26.81
- 5‑HT 1.0 → α_mean≈0.276; open_rate≈0.215; energy_post≈24.95

Interpretation
- Divisive gain works as intended: increasing 5‑HT monotonically reduces residual influence (α) and halting open‑rate, with a commensurate drop in post‑gate energy. Selectivity/logits are preserved (no code‑path changes), matching “amplitude‑as‑attention” modulation.

Artifacts
- Per‑level reports at `_telem_5ht/level_*/{metrics_summary.json,report.md,sparks/*.svg}`.

---

## 2025-08-29 — PLNN: LLR Mask → Memory (Gated Rehearsal/Writes)

- Scope: In external repo `/Volumes/stuff/Projects/AI/LNNDemo/phonological_loop` wired the statistical LLR noise mask into the Phonological Loop pipeline to (a) gate memory writes and (b) drive rehearsal selection.

Changes (external)
- models/statistical_noise_filter.py: forward returns `(filtered, mask)` to expose the soft salience mask.
- models/memory.py: `forward(features, mask=None)`; write `s·features` to the ring buffer (s = mean(mask)∈[0,1]); if `s>0.5`, set rehearsal to current features; else use most‑recent window.
- models/phonological_loop_classifier.py: added `LogDomainNoiseSuppression`; forward uses `filtered, mask = noise_filter(analytic_feats)` and passes `memory(filtered, mask=mask)`.
- Quick‑check: `llr_mask_quickcheck.py` shows mask_mean clean≈0.515 → noisy≈0.257 (sanity evidence that mask suppresses noise).

Why (biophysical + engineering)
- Aisbett‑robust statistics give unbiased salience under narrowband noise → principled “carry detector”.
- Preserve→Gate: decay keeps value, salience gates influence at write and in rehearsal (value‑preserving, influence‑modulating).
- Time Writes The Story: rehearsal acts as the “carry channel” selector for the clip‑and‑carry mechanism.
- Vector‑DB of masked waves: memory is a short decayed store of masked components; the classifier reads a composed state.

Notes
- S4 here runs with L=1 (acts as a projector). PyKeOps CUDA backends are not configured; full forward not run in this env; the quickcheck isolates the mask behavior.

## Next (tracked here)
- Add a small aggregator to compute rolling means and spark-lines from `runs/*/*.csv`.
- Optional per-block cube gating flag + update budget tests.
- LM trainer with masking + proper vocab head and ponder.

---

## 2025-08-29 — Citation Sea + Provenance Index

- Added refs/citation_sea.md: curated bibliography grounding dendritic comb‑filter architecture across biophysics, oscillations, information theory, computation, and systems.
- Updated origin paper (40_ukm_origin_paper.md) to reference the citation sea and to note provenance/IP.
- Added 41_provenance_and_ip.md to record authorship and adaptation notes (1980s Australian intelligence documents per author’s statement).

Rationale
- Ensure every architectural claim has bidirectional anchors: biology ↔ math ↔ engineering. Keep references public and traceable.
