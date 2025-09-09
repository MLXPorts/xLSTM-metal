
"""
Apply project-local fixups to docs/mlx_reference/* copied from MetalFaiss.

- Prepend a short note indicating they were ported and adapted for xLSTM MLX.
- Update common path references to local equivalents.
"""
from __future__ import annotations
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "mlx_reference"

HEADER = (
    "<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->\n\n"
)

REWRITES = [
    ("python/metalfaiss/utils/streams.py", "tools/mlx_streams.py"),
    ("python/metalfaiss/faissmlx/kernels/gemm_kernels.py", "mlx_fast_kernels/gemm_kernels.py"),
    ("docs/mlx/", "docs/mlx_reference/"),
]

def fix_file(p: pathlib.Path) -> None:
    txt = p.read_text(encoding="utf-8")
    # Prepend header only once
    if not txt.startswith("<!-- Note: Ported from MetalFaiss"):
        txt = HEADER + txt
    for old, new in REWRITES:
        txt = txt.replace(old, new)
    p.write_text(txt, encoding="utf-8")

def main() -> None:
    """The main function of the script."""
    for p in sorted(DOCS.glob("*")):
        if not p.is_file():
            continue
        fix_file(p)

if __name__ == "__main__":
    main()

