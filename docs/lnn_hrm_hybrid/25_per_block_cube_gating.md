# Per‑Block Cube Gating — Design Draft

Goal: Optionally attach MemoryCube+α‑gate to every xLSTM block boundary (not just the stack boundary) with a capped update budget per block, while preserving stability and reproducibility.

## API (proposed)
- `HRMXLSTM(..., per_block_cube: bool = False, max_updates_per_block: int | None = None)`
- When enabled, each block gets a `CubeGatedBlock` instance; writes obey Z5 and per‑block budget.

## Scheduling
- Keep global Z5 envelope. A write is allowed only if:
  - `slot == 4` and `updates_this_block_in_cycle < max_updates_per_block` (if set).
- Track per‑block counters; reset counters on cycle rollover (4→0 carry).

## Telemetry
- Add per‑block: `{alpha_mean_i, conf_mean_i, commits_i}`.
- Aggregate: running means, open rate, update budget utilization.

## Safety
- Value preserving: keep pre‑gate states per block for audits.
- Budget checks: deny writes when UMA/memory pressure high.

## Tests
- Unit: budget respected; only boundary commits counted; counters reset on rollover.
- Integration: multi‑block model improves conf on second pass; no shape drift.

## Migration Plan
1) Add flag + counters; plumb allow_commit mask into blocks.
2) Extend telemetry and unit tests.
3) Update Research Journal with runs comparing boundary‑only vs per‑block.

