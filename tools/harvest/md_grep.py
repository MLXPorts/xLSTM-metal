#!/usr/bin/env python3
import argparse
from pathlib import Path
import re


def grep_with_context(path: Path, patterns, before: int, after: int, ignore_case: bool):
    flags = re.IGNORECASE if ignore_case else 0
    res = []
    text = path.read_text(errors='ignore')
    lines = text.splitlines()
    n = len(lines)
    regs = [re.compile(p, flags) for p in patterns]
    for i, line in enumerate(lines, start=1):
        if any(r.search(line) for r in regs):
            start = max(1, i - before)
            end = min(n, i + after)
            res.append((i, lines[start - 1:end], start, end))
    return res


def main():
    ap = argparse.ArgumentParser(description='Markdown grep with context and line numbers')
    ap.add_argument('file', type=str)
    ap.add_argument('-e', '--expr', action='append', required=True, help='regex to search (repeatable)')
    ap.add_argument('-A', '--after', type=int, default=5)
    ap.add_argument('-B', '--before', type=int, default=5)
    ap.add_argument('-i', '--ignore-case', action='store_true')
    args = ap.parse_args()
    path = Path(args.file)
    hits = grep_with_context(path, args.expr, args.before, args.after, args.ignore_case)
    for i, block, start, end in hits:
        print(f"-- L{i} (context {start}–{end}) --")
        for j, ln in enumerate(block, start=start):
            print(f"{j:>6}: {ln}")
        print()


if __name__ == '__main__':
    main()

