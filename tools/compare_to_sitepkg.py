
"""
Compare our in-repo "official" mirror against the installed site-packages xlstm code.

Compares files under:
- A: xlstm_official_full/xlstm_large
- B: site-packages xlstm/xlstm_large

Outputs:
- Summary counts (added/removed/modified/same)
- Per-file status, and for modified files a short diffstat (#changed lines)
"""
from __future__ import annotations
import sys
import os
import hashlib
import difflib
from pathlib import Path


def read_text(p: Path) -> list[str]:
    try:
        return p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def digest(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def collect(base: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in base.rglob("*.py"):
        rel = p.relative_to(base).as_posix()
        files[rel] = p
    return files


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ours_base = repo_root / "xlstm_official_full" / "xlstm_large"
    if not ours_base.exists():
        print(f"error: not found: {ours_base}")
        return 1

    # find site-packages xlstm
    try:
        import xlstm as xlstm_pkg  # type: ignore
        pkg_root = Path(xlstm_pkg.__file__).resolve().parent
        theirs_base = pkg_root / "xlstm_large"
        if not theirs_base.exists():
            print(f"error: installed xlstm has no xlstm_large at: {theirs_base}")
            return 1
    except Exception as e:
        print("error: could not import xlstm from site-packages:", e)
        return 1

    ours = collect(ours_base)
    theirs = collect(theirs_base)

    all_keys = sorted(set(ours.keys()) | set(theirs.keys()))
    added = []
    removed = []
    modified = []
    same = []

    for k in all_keys:
        pa = ours.get(k)
        pb = theirs.get(k)
        if pa and not pb:
            added.append(k)
            continue
        if pb and not pa:
            removed.append(k)
            continue
        # both exist
        if digest(pa) == digest(pb):
            same.append(k)
        else:
            modified.append(k)

    print("Compare xlstm_official_full/xlstm_large vs site-packages xlstm/xlstm_large")
    print(f"  same: {len(same)}  modified: {len(modified)}  added: {len(added)}  removed: {len(removed)}")
    if added:
        print("\nAdded:")
        for k in added:
            print("  +", k)
    if removed:
        print("\nRemoved:")
        for k in removed:
            print("  -", k)
    if modified:
        print("\nModified (diffstats):")
        for k in modified:
            pa = ours[k]
            pb = theirs[k]
            a = read_text(pa)
            b = read_text(pb)
            diff = list(difflib.unified_diff(b, a, fromfile=str(pb), tofile=str(pa), n=0))
            # Count changed hunks roughly
            changes = sum(1 for line in diff if line and line[0] in "+-")
            print(f"  ~ {k}  (~{changes} changed lines)")

    # Optionally write a full diff file for inspection
    out = repo_root / "tools" / "_compare_full.diff"
    with out.open("w", encoding="utf-8") as f:
        for k in modified:
            pa = ours[k]
            pb = theirs[k]
            a = read_text(pa)
            b = read_text(pb)
            for line in difflib.unified_diff(b, a, fromfile=str(pb), tofile=str(pa), n=3):
                f.write(line + "\n")
    print(f"\nWrote full unified diff to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

