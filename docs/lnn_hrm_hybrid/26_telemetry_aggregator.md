# Telemetry Aggregator — Design Draft

Purpose: Read CSV/JSONL logs from `runs/*/*.{csv,jsonl}`, compute rollups (means, EWMA trends), and emit lightweight plots/snippets for docs.

## CLI (proposed)
- `python -m tools.telem.aggregate --glob 'runs/*/*.csv' --out docs/lnn_hrm_hybrid/_telem`
- Options: `--metrics alpha_mean,conf_mean,act_prob_mean,act_open_rate,energy_pre_gate,energy_post_gate,loss,ce,ponder`

## Outputs
- `metrics_summary.json`: last, mean, std for each metric.
- `sparks/*.svg`: sparkline per metric (fixed-size inlineable SVGs).
- `report.md`: embeds summary + images; include in Research Journal.

## Implementation Notes
- Use `pandas` if available; fall back to csv module.
- SVGs via `matplotlib` or minimal inline SVG generator.
- Determinism: run seeded; order by `step`.

## Tests
- Parse a tiny synthetic CSV; verify summary stats.
- Ensure SVGs are generated and <20KB each.

