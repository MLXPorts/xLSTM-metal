# E-PLNN-LLR-GATE — Phonological Loop: LLR Mask → Memory Gating

- Date: 2025-08-29
- Repo: /Volumes/stuff/Projects/AI/LNNDemo/phonological_loop
- Change: integrate LogDomainNoiseSuppression; gate memory writes and rehearsal by salience

## Hypothesis
A statistical LLR mask derived from analytic-signal features will reduce salience under noise and improve which frames are rehearsed/written to memory.

## Methods
- Code: models/statistical_noise_filter.py returns `(filtered, mask)`; models/memory.py now `forward(features, mask=None)` and scales writes by mean mask; if `s>0.5` choose current frame for rehearsal.
- Quickcheck script: `llr_mask_quickcheck.py` (synthetic clean vs noisy features)

## Data
- mask_mean (clean) ≈ 0.515
- mask_mean (noisy) ≈ 0.257

## Results
- Salience drops under added noise as expected; mask is a plausible carry detector for rehearsal and write gating.

## Missteps / Caveats
- Quickcheck only; full end‑to‑end eval blocked here due to PyKeOps CUDA backend in local S4 path.

## Next
- End‑to‑end PLNN run with mask on/off ablation; measure classification accuracy and stability under stronger noise.
- Consider wiring the same mask concept optionally in HRM as a pre‑MemoryCube gate (#11).

## Links
- Issues: #11 (LLR mask in HRM)
- Files: models/statistical_noise_filter.py, models/memory.py, models/phonological_loop_classifier.py, llr_mask_quickcheck.py

