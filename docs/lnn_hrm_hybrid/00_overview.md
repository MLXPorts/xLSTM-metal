# SolaceCore HRM+ for UKM — Overview

This scaffold integrates a two–timescale hybrid into the UKM repo:

- Fast liquid module (continuous-time recurrent cell) for rapid, local updates.
- Slow planner/context via `SpiralAttention` and UKM spiral features.
- Per-block Memory Cube (cosine top‑K) with confidence-gated blending.
- Telemetry hooks (alpha gate mean, retrieval confidence) for audits.

Code entrypoints:
- `ukm.hrm.transformer_lnn.TransformerLNNHybrid` — end-to-end block.
- `ukm.hrm.memory_cube.MemoryCube` — content-addressable memory.
- `ukm.hrm.liquid_time_constant.LiquidTimeConstantCell/LiquidBlock` — fast module.
- `ukm.hrm.cube_gated_block.CubeGatedBlock` — residual blend and gating.

Example: `examples/transformer_lnn_example.py` (MPS-ready, <3s on AS Macs).


Additional references (formal + bio mapping):

- 22_fractal_diffusion_formalism.md — full formal model over ℚ with proofs
- 23_bio_rall_hcn_mapping.md — dendritic/Rall/HCN mapping to engineering invariants
