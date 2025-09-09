
"""
Emit a manifest of the upstream xLSTM mirror directory.
Writes to stdout in a simple, sorted list of relative paths.
"""
from __future__ import annotations
import sys
from pathlib import Path

def main(base: str = "xlstm_official") -> int:
    """Prints a manifest of the upstream xLSTM mirror directory.

    This function prints a manifest of the upstream xLSTM mirror directory to
    stdout. The manifest is a sorted list of relative paths.

    Args:
        base (str, optional): The base directory of the xLSTM mirror. Defaults
            to "xlstm_official".

    Returns:
        0 on success, 1 on failure.
    """
    root = Path(base)
    if not root.exists():
        print(f"error: {root} not found", file=sys.stderr)
        return 1
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    for f in files:
        print(f)
    return 0

if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:])))

