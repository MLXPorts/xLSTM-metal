#!/usr/bin/env python3
"""
Telemetry Aggregator

Reads CSV logs (glob) and emits:
- metrics_summary.json: last/mean/std/min/max/count per metric
- sparks/<metric>.svg: sparkline per metric
- report.md: markdown summary embedding images

Usage:
  python -m tools.telem.aggregate --glob 'runs/*/*.csv' --out docs/lnn_hrm_hybrid/_telem \
      --metrics alpha_mean,conf_mean,act_prob_mean,act_open_rate,energy_pre_gate,energy_post_gate,loss,ce,ponder
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate telemetry CSV logs and emit summary + sparkline SVGs")
    ap.add_argument("--glob", default="runs/*/*.csv", help="Glob of CSV files")
    ap.add_argument("--out", default="docs/lnn_hrm_hybrid/_telem", help="Output directory")
    ap.add_argument("--metrics", default="", help="Comma-separated metrics to include; default: auto-detect numeric fields")
    ap.add_argument("--ewma", type=float, default=0.2, help="EWMA alpha for smoothing (0..1)")
    ap.add_argument("--limit", type=int, default=400, help="Limit to last N points per metric (0 = all)")
    return ap.parse_args()


def is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def read_csv_series(paths: List[Path], metrics: List[str] | None) -> Tuple[Dict[str, List[float]], List[str]]:
    series: Dict[str, List[float]] = {}
    fields_all: List[str] = []
    for p in paths:
        with p.open("r", newline="") as f:
            r = csv.DictReader(f)
            if not fields_all:
                fields_all = list(r.fieldnames or [])
            for row in r:
                # choose metrics dynamically if not provided
                if metrics is None:
                    metrics = [k for k, v in row.items() if k not in ("step", "ts", "trace_hash") and is_float(v)]
                for m in metrics:
                    v = row.get(m, None)
                    if v is None or not is_float(v):
                        continue
                    series.setdefault(m, []).append(float(v))
    return series, fields_all


@dataclass
class Summary:
    last: float
    mean: float
    std: float
    min: float
    max: float
    count: int


def summarize(xs: List[float]) -> Summary:
    import math
    n = len(xs)
    if n == 0:
        return Summary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(max(var, 0.0))
    return Summary(xs[-1], mu, std, min(xs), max(xs), n)


def ewma(xs: List[float], alpha: float) -> List[float]:
    if not xs:
        return []
    out = []
    s = xs[0]
    out.append(s)
    for x in xs[1:]:
        s = alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def spark_svg(xs: List[float], width: int = 240, height: int = 60, margin: int = 4, stroke: str = "#4a90e2") -> str:
    if not xs:
        return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'></svg>"
    if len(xs) == 1:
        xs = xs * 2
    w = width - 2 * margin
    h = height - 2 * margin
    xmin, xmax = 0, len(xs) - 1
    ymin, ymax = min(xs), max(xs)
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0
    def xmap(i: int) -> float:
        return margin + w * (i - xmin) / (xmax - xmin)
    def ymap(v: float) -> float:
        return margin + h * (1 - (v - ymin) / (ymax - ymin))
    pts = " ".join(f"{xmap(i):.2f},{ymap(v):.2f}" for i, v in enumerate(xs))
    path = f"<polyline fill='none' stroke='{stroke}' stroke-width='1.5' points='{pts}'/>"
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>" \
           f"{path}</svg>"


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in sorted(Path().glob(args.glob))]
    if not paths:
        print(f"[telem] No CSV files matched {args.glob}")
        return
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
    series, fields = read_csv_series(paths, metrics)
    if metrics is None:
        metrics = sorted(series.keys())
    # apply limit
    lim = int(args.limit)
    data_limited = {m: (xs[-lim:] if lim > 0 else xs) for m, xs in series.items()}
    # summaries
    summ = {m: summarize(xs).__dict__ for m, xs in data_limited.items()}
    (outdir / "metrics_summary.json").write_text(json.dumps(summ, indent=2))
    # sparks
    sparks_dir = outdir / "sparks"
    sparks_dir.mkdir(parents=True, exist_ok=True)
    for m in metrics:
        xs = data_limited.get(m, [])
        if not xs:
            continue
        ys = ewma(xs, args.ewma)
        svg = spark_svg(ys)
        (sparks_dir / f"{m}.svg").write_text(svg)
    # report
    lines = ["# Telemetry Summary", "", f"Files: {len(paths)}", ""]
    for m in metrics:
        s = summ.get(m)
        if not s:
            continue
        lines.append(f"## {m}")
        lines.append(f"- last: {s['last']:.6g}; mean: {s['mean']:.6g} ± {s['std']:.3g}; min–max: {s['min']:.6g}–{s['max']:.6g}; n={s['count']}")
        svg_rel = f"sparks/{m}.svg"
        lines.append(f"![]({svg_rel})\n")
    (outdir / "report.md").write_text("\n".join(lines))
    print(f"[telem] Wrote {outdir}/metrics_summary.json and {outdir}/report.md")


if __name__ == "__main__":
    main()

