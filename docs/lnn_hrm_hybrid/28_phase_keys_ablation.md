# Phase-Key Features — Ablations

Goals: Validate whether phase features (fast/mid/slow + Z5 one-hot) improve retrieval/gating vs. base keys; explore a small MLP over phase features.

## Toggles
- `fuse_phase_keys=True|False` (existing)
- `phase_mlp_hidden` (proposed): None → selection; int → small MLP

## Metrics
- conf_mean uplift after first commit
- CE/ponder trends on tiny trainer

## Tests
- Shapes stable across toggles
- conf_mean increases after commit with and without phase

