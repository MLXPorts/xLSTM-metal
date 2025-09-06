#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_LARGE_DIRS = {"runs/", "xlstm_7b_model/"}
LARGE_FILE_BYTES_FAIL = int(os.environ.get("XLSTM_POLICY_LARGE_FILE_BYTES", str(25 * 1024 * 1024)))
ALLOW_LARGE = os.environ.get("XLSTM_POLICY_ALLOW_LARGE_FILES", "0") == "1"
OVERRIDE = os.environ.get("XLSTM_POLICY_OVERRIDE", "0") == "1"

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".mypy_cache", ".pytest_cache"}
TEST_DIR_HINTS = {"tests", "test", "examples", "example", "demos", "demo"}

RE_MOCK_IMPORT = re.compile(r"\b(unittest\.mock|from\s+unittest\s+import\s+mock|import\s+mock)\b")
RE_MOCK_SYMBOL = re.compile(r"\b(MagicMock\s*\(|AsyncMock\s*\(|Mock\s*\(|patch\s*\()")
# “Weasel words” that often indicate non-production shortcuts or disclaimers.
# Keep this list tight; match whole words/phrases to reduce false positives.
RE_PROHIBITED_WORDS = re.compile(
    r"\b(" 
    r"simplified|for\s+simplicity|toy|placeholder|dummy(?:\s+implementation)?|"
    r"fake(?:\s+implementation)?|approximate|rough\s+sketch|"
    r"we\s+won't\s+implement|will\s+not\s+implement|we\s+will\s+not\s+implement|"
    r"simulate|pretend"
    r")\b",
    re.IGNORECASE,
)
RE_RAY_INIT = re.compile(r"ray\.init\(")
RE_RAY_SHUTDOWN = re.compile(r"ray\.shutdown\(")
RE_NEAR_COPY_NAME = re.compile(r"(_copy|_new|_alt|_tmp|copy|backup)\.[a-zA-Z0-9]+$")

# Disallow Swift or Xcode artifacts in this repo
SWIFT_BAD_EXT = ".swift"
DISALLOWED_MAC_PROJ_SUFFIXES = (".xcodeproj", ".xcworkspace")
DISALLOWED_MAC_FILES = {"Package.swift"}


def is_in_dirs(path: Path, names) -> bool:
    parts = set(path.parts)
    return any(n in parts for n in names)


def staged_files() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "diff", "--name-only", "--cached"], cwd=ROOT).decode()
        files = [ROOT / p for p in out.split() if p.strip()]
        return [p for p in files if p.exists()]
    except Exception:
        return []


def all_repo_files() -> list[Path]:
    files = []
    for p in ROOT.rglob("*"):
        if p.is_dir():
            if p.name in SKIP_DIRS:
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            continue
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        files.append(p)
    return files


def should_scan_text(p: Path) -> bool:
    if p.suffix in {".py", ".md", ".sh", ".txt", ".toml", ".yaml", ".yml"}:
        return True
    return False


def policy_check(paths: list[Path], soft_enforce: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        # hard disallow: Swift / Xcode artifacts
        if p.suffix == SWIFT_BAD_EXT:
            errors.append(f"SWIFT_FILE: {rel} should not be present in this repo")
        if any(rel.endswith(sfx) for sfx in DISALLOWED_MAC_PROJ_SUFFIXES):
            errors.append(f"XCODE_ARTIFACT: {rel} should not be present in this repo")
        if p.name in DISALLOWED_MAC_FILES:
            errors.append(f"SWIFT_PACKAGE: {rel} should not be present in this repo")
        # large file gate
        try:
            if p.is_file():
                size = p.stat().st_size
                if size >= LARGE_FILE_BYTES_FAIL and not ALLOW_LARGE:
                    if not any(rel.startswith(d) for d in ALLOWED_LARGE_DIRS):
                        errors.append(f"LARGE_FILE: {rel} is {size/1e6:.1f} MB (> {LARGE_FILE_BYTES_FAIL/1e6:.1f} MB)")
        except Exception:
            pass

        # name anti-patterns
        if RE_NEAR_COPY_NAME.search(p.name):
            errors.append(f"NEAR_COPY_NAME: suspicious filename '{rel}' (avoid *_copy/_new/_alt)")

        if not should_scan_text(p):
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue

        in_tests = is_in_dirs(p, TEST_DIR_HINTS)
        is_py = p.suffix == ".py"

        # Skip scanning this policy script itself to avoid self-flagging
        if p.resolve() == (ROOT / "scripts/lint/check_repo_policy.py").resolve():
            continue

        if is_py and not in_tests:
            if RE_MOCK_IMPORT.search(text) or RE_MOCK_SYMBOL.search(text):
                msg = f"MOCK_IN_PROD: mock usage in prod path {rel}"
                (errors if soft_enforce else warnings).append(msg)
            if RE_PROHIBITED_WORDS.search(text):
                msg = f"WEASEL_WORD: discouraged wording in {rel} (simplified/toy/placeholder/dummy/fake)"
                (errors if soft_enforce else warnings).append(msg)

            if RE_RAY_INIT.search(text) and not RE_RAY_SHUTDOWN.search(text):
                errors.append(f"RAY_SHUTDOWN_MISSING: {rel} calls ray.init but not ray.shutdown")

    if errors or warnings:
        if OVERRIDE:
            print("Policy violations (override enabled):")
            if warnings:
                print("Warnings:\n" + "\n".join(warnings))
            if errors:
                print("Errors:\n" + "\n".join(errors))
            return 0
        if warnings:
            print("Policy warnings:")
            print("\n".join(" - " + w for w in warnings))
        if errors:
            print("Policy errors:")
            print("\n".join(" - " + e for e in errors))
            return 2
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-staged", action="store_true", help="Check only staged files (for pre-commit)")
    args = ap.parse_args()
    paths = staged_files() if args.only_staged else all_repo_files()
    soft_enforce = bool(args.only_staged)
    sys.exit(policy_check(paths, soft_enforce))


if __name__ == "__main__":
    main()
