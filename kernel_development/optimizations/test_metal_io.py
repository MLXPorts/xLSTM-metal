
"""
GPU-only IO test for Metal buffer mapping:
- Compiles the shared shader and runs memcpy_kernel to copy an MPS tensor.
- Verifies the output exactly matches the input.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.cpp_extension import load


def read_shader() -> str:
    """

    :return:
    """
    root = Path(__file__).resolve().parents[1]
    for _ in range(6):
        met = (root / "kernels/metal/shaders/mlstm_kernels.metal").resolve()
        if met.exists():
            return met.read_text()
        arc = (root / "research_archive/metal_prototypes/kernels_metal/shaders/mlstm_kernels.metal").resolve()
        if arc.exists():
            return arc.read_text()
        root = root.parent
    raise FileNotFoundError("mlstm_kernels.metal not found")


def main() -> int:
    """

    :return:
    """
    assert torch.backends.mps.is_available(), "MPS required"
    mm = _find_backend_source()
    assert mm is not None, "Metal backend .mm not found"
    mod = load(name="mlstm_metal_backend", sources=[str(mm)],
               extra_ldflags=["-framework", "Metal", "-framework", "Foundation"])
    src = read_shader()

    def check(shape):
        """

        :param shape:
        :return:
        """
        x = torch.randn(*shape, device="mps", dtype=torch.float32)
        y = mod.metal_memcpy_with_source(x, src)
        diff = (x - y).abs().max().item()
        print(f"  shape={shape} diff={diff:.3e}")
        return diff

    print("Testing Metal memcpy...")
    diffs = []
    for shp in [(16,), (3,4), (2,3,5), (1,2,8,8), (2,2,4,4)]:
        diffs.append(check(shp))
    ok = all(d == 0.0 for d in diffs)
    print("Result:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
