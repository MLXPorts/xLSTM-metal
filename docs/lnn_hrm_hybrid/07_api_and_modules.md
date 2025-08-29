# API and Modules

- `TransformerLNNHybrid(input_dim, hidden_dim, seq_len, cube_capacity, blocky_levels=None)`
  - `forward(x, update_memory=True) -> dict {output, alpha_mean, conf}`
  - Uses: `LiquidBlock` → `SpiralAttention` → `CubeGatedBlock`.

- `MemoryCube(key_dim, value_dim, capacity=1024, topk=8, fuse_weight=0.3)`
  - `query(q, spike_key=None, topk=None, temperature=0.1)`
  - `update(key, value, spike_key=None)`
  - `spike_comb(x, bins=32, threshold=0.0)`

- `LiquidTimeConstantCell(input_dim, hidden_dim, dt_max=0.2, tau_init=1.5, blocky_levels=None)`
  - Discretized continuous-time update with optional blocky activation.

- `CubeGatedBlock(dim, cube, hidden=256)`
  - Residual mix of features with retrieved memory; returns telemetry.

## Origins at a Glance
- HRMXLSTM wrapper → Invariants (Preserve→Gate, Z5, Budget) mapped at 22 §10–11; provenance map 32.
- MemoryCube/CubeGatedBlock → Preserve→Gate + remainder/phase keys (22 §10); bio comb mapping (23).
- Scheduler → base‑5 carry policy (22; mermaid sketch); tests ensure boundary‑only updates.
- ACTHaltingHead/PonderTrainer → budget discipline with halting analogue (05; 22 §11).
- Telemetry (energy/logger/trace) → stability and determinism (06; 22 §11; Research Journal 24).

## Neuromodulator Inputs (Prototype)
- `HRMXLSTM(...).forward(x, times, mod_5ht=None)`
  - `mod_5ht` is a per‑token scalar in [0,∞): higher → divisive gain on residual influence and slightly higher halting threshold.
  - Telemetry adds `mod_5ht_mean`, `gain_5ht_mean`.
- Roadmap: `mod_ach`, `mod_da`, `mod_ne` with small bounded transforms (gain/sharpening, f0 bias, SNR), aligned to “Neuromodulator CPU and Timescales” in the origin paper.
