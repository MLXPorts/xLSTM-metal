#!/usr/bin/env python
import argparse
import sys
import os

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
try:
    import emberlint  # noqa: E402
except ModuleNotFoundError:
    # Allow invocation when cwd is project root and tools not on path
    sys.path.insert(0, os.path.abspath(HERE))
    import emberlint  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Run emberlint on a path and summarize precision casts and tensor conversions")
    ap.add_argument("path", help="File or directory to analyze")
    args = ap.parse_args()

    p = args.path
    if os.path.isdir(p):
        results = emberlint.analyze_directory(p, exclude_dirs=["__pycache__", ".venv", "venv", ".git"])
    else:
        results = [emberlint.analyze_file(p)]

    total_casts = sum(len(r.get("precision_casts", [])) for r in results)
    total_convs = sum(len(r.get("tensor_conversions", [])) for r in results)

    print(f"Analyzed {len(results)} file(s)")
    print(f"Precision-reducing casts (float()/int() on tensors): {total_casts}")
    print(f"Tensor conversions (potential .item() / backend moves): {total_convs}")

    if total_casts or total_convs:
        print("\nDetails:")
        for r in results:
            if r.get("precision_casts"):
                print(f"\n{r['file']}")
                for c in r["precision_casts"]:
                    print(f"  CAST {c['type']} at {c['location']}")
            if r.get("tensor_conversions"):
                print(f"\n{r['file']}")
                for c in r["tensor_conversions"]:
                    print(f"  CONV {c['type']} at {c['location']}")

if __name__ == "__main__":
    main()
