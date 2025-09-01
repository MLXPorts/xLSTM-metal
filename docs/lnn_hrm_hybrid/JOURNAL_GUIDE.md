# Research Journal Guide

A lightweight, repeatable structure for recording experiments, figures, data tables, missteps, hypotheses, and conclusions. Each entry is a standalone Markdown file under `docs/lnn_hrm_hybrid/journal/` and participates in the index.

## Entry Template
- Title: E-YYYYMMDD-KEY — short, unique mnemonic
- Authors/Env: host, device, Python/torch versions, seeds
- Hypothesis: what you expect and why (1–3 lines)
- Methods: scripts, flags, datasets, configs (exact commands)
- Data: tables (copy from metrics_summary.json via `hrm-telem-aggregate` or `tools/journal/render_summary.py`)
- Figures: PNG/SVG paths under `docs/.../_telem*/...`
- Results: observations tied to hypothesis
- Missteps: dead ends, surprises, caveats
- Next: concrete follow-ups, linked to issues
- Links: PRs, issues, runs, source files

## Helpers
- Aggregate telemetry to figures and a report:
  - `hrm-telem-aggregate --glob 'runs/*/*.csv' --out docs/lnn_hrm_hybrid/_telem`
- Render a quick Markdown table from a metrics_summary.json:
  - `python -m tools.journal.render_summary docs/.../metrics_summary.json --metrics alpha_mean,act_open_rate,energy_post_gate`

## Naming & Provenance
- Use deterministic seeds and record `trace_hash` where applicable.
- Include absolute or repo‑relative paths to artifacts.
- Keep entries short; link to raw logs and reports.

