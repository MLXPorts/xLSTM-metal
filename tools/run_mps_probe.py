
import sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]

def main():
    assert torch.backends.mps.is_available(), "MPS required"
    src = str(ROOT / "mps_probe/mps_probe.mm")
    mod = load(
        name="xlstm_mps_probe",
        sources=[src],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=True,
    )
    x = torch.randn(64, device='mps', dtype=torch.float32)
    y = torch.ops.xlstm_mps_probe.debug_memcpy(x)
    diff = (x - y).abs().max().item()
    print(f"debug_memcpy max diff: {diff:.3e}")

if __name__ == "__main__":
    main()

