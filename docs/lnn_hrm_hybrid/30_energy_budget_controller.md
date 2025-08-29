# Energy Budget Controller — Design

Purpose: Move from audit-only to active control of energy budgets at the HRM wrapper boundary (and per-block when enabled).

## Signals
- `energy_pre_gate, energy_post_gate, ΔE`, running means
- Optional: gradient norms, α statistics

## Actions
- Clamp α range when ΔE exceeds thresholds
- Trigger cube write denial on sustained overload
- Log reason codes in telemetry

## API
- `HRMXLSTM(..., energy_guard: Optional[dict])` e.g., `{max_delta_mean: 0.1, window: 64}`

## Tests
- Synthetic overload induces α clamp; ΔE trend recovers
- Telemetry contains guard actions and reason codes

