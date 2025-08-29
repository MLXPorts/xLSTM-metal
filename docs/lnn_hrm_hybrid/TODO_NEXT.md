# TODO / Next — Harvest + Engineering

This consolidates near‑term tasks with exact anchors into your large markdown source and the codebase.

## Harvest From “Jul 15, 2025 … Markdown Content.md” (line anchors)
- Serotonin Q&A intro: L5372–5381 — “tune the constants” framing.
- 5‑HT effects table: L5393–5401 — receptors (5‑HT1A/2A/4,7) → ΔV½, Δf0, Δσ, ΔImax.
- Implementation hook: L5458 — replace `neuromod[…]` with real‑time 5‑HT concentrations.
- Experiments: L5464–5468 — “serotonin state‑machine”; “mood log‑bit.”
- Pillars summary: L5916–5919 — “Neuromod‑as‑parameter‑shift (ACh, DA, 5‑HT, NE).”
- Scope of shifts: L6048–6049; L6238–6240 — neuromod shifts act only on {ΔV½, f0′, σ′, M}.
- Prediction: L6310–6313 — 5‑HT2A agonist broadens BK σ by ~30% (chirp).
- Timescales (CPU): L5836+ — ms gates; seconds→minutes neuromod; hours→years structure.
- Lifespan anchor: L3115–3117 — DNA locks hyper‑parameters “80 years vs 80 ms.”
- Evolution anchor: L3066–3070 — millions→billions of years hyperparameterization.
- Scale‑free bridge: L1606–1609 — attoseconds→gigayears “superpose→threshold.”

Planned actions
- Paper (40_ukm_origin_paper.md):
  - Import 5‑HT table + “state‑machine/mood log‑bit” with anchors; add references.
  - Add ACh/DA/NE sections mirroring 5‑HT (gain/sharpening, reward/f0 bias, SNR/NE).
  - Fold predictions (σ widening; commit window clustering) into “Falsifiable tests.”
- Figures: run 5‑HT gain sweep and add α/open‑rate/energy plots; add probe panels.

## Engineering
- Telemetry aggregator (26): CSV/JSONL → summary.json + SVG sparks + report.md.
- Spike‑onset/triangle recon (33): add raster, tri/bi‑exp recon; vector‑DB demo.
- Order‑aware keys: rank‑order (OeSNN‑style) or permutation embeddings; ablations.
- Per‑block cube gating (25): flag + per‑block update budgets + tests.
- Modulators: extend beyond 5‑HT (ACh/DA/NE) with bounded transforms and telemetry.
- MemoryCube: merge‑by‑similarity + TTL/LRU + UMA backpressure (29).
- Energy guard (30): active clamp + reason codes + tests.
- Full LM trainer (27): proper head, masking, eval; logging integration.

## Tests & CI (31)
- MPS integration: long‑run determinism (trace_hash replay); multi‑block stress.
- CPU suite: harvest tools, MemoryCube, order‑keys, anomaly score.

## Tools
- Harvest script: `tools/harvest/md_grep.py` (grep‑like with context for anchors); build index of matches for serotonin/ACh/DA/NE/“months”/“years”.

## Notes
- Keep anchors stable: refer to the original file name + line ranges above.
- When quoting, prefer short paraphrases; store full excerpts locally only if license permits.

