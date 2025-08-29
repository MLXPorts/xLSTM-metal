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

