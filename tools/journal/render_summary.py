#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path')
    ap.add_argument('--metrics', default='')
    args = ap.parse_args()
    p = Path(args.json_path)
    data = json.loads(p.read_text())
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()] or list(data.keys())
    print('| metric | last | mean | std | min | max | n |')
    print('|:--|--:|--:|--:|--:|--:|--:|')
    for m in metrics:
        s = data.get(m)
        if not s: continue
        print(f"| {m} | {s['last']:.6g} | {s['mean']:.6g} | {s['std']:.3g} | {s['min']:.6g} | {s['max']:.6g} | {s['count']} |")

if __name__ == '__main__':
    main()
