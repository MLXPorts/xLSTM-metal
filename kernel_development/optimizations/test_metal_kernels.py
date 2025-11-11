"""
Quick Metal kernel self-test (MPS only).
Runs soft_cap and a single mLSTM step on tiny tensors to validate JIT/build.
"""
import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    """Runs a self-test of the Metal kernels.

    This function runs a self-test of the Metal kernels to validate that they
    can be JIT-compiled and executed on the MPS backend.

    Returns:
        0 on success, 1 on failure.
    """
    if not torch.backends.mps.is_available():
        print("MPS not available; this test requires an Apple GPU.")
        return 1
    device = "mps"

    print("Testing Metal soft_cap...")
    from mlstm_kernels.torch.metal.softcap import metal_soft_cap
    x = torch.randn(32, device=device, dtype=torch.float32)
    y = metal_soft_cap(x, 5.0)
    print("  soft_cap ok:", y.shape, y.dtype, y.device)

    print("Testing Metal mLSTM step...")
    from mlstm_kernels.torch.recurrent.metal.step import mlstm_recurrent_step__metal_fw
    B, NH, DHQK, DHHV = 1, 2, 8, 8
    q = torch.randn(B, NH, DHQK, device=device)
    k = torch.randn(B, NH, DHQK, device=device)
    v = torch.randn(B, NH, DHHV, device=device)
    i = torch.randn(B, NH, device=device)
    f = torch.randn(B, NH, device=device)
    C = torch.zeros(B, NH, DHQK, DHHV, device=device)
    N = torch.zeros(B, NH, DHQK, device=device)
    M = torch.zeros(B, NH, device=device)
    h, (C2, N2, M2) = mlstm_recurrent_step__metal_fw(C, N, M, q, k, v, i.unsqueeze(-1), f.unsqueeze(-1))
    print("  step ok:", h.shape, C2.shape, N2.shape, M2.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
