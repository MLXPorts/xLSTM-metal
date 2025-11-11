"""
Parity test for Metal step vs native step and native_sequence vs native_sequence__metal.
Runs on tiny shapes and prints max abs/rel errors.
"""
import math
import sys
from pathlib import Path
import torch

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlstm_kernels.torch.recurrent.native_step import mlstm_recurrent_step__native_fw
from mlstm_kernels.torch.recurrent.metal.compiled import mlstm_recurrent_step__metal_fw
from mlstm_kernels.torch.recurrent.native_sequence import mlstm_recurrent_sequence__native_fw
from mlstm_kernels.torch.recurrent.__init__ import registry_sequence as _seq_reg


def max_err(a: torch.Tensor, b: torch.Tensor):
    """

    :param a:
    :param b:
    :return:
    """
    diff = (a - b).abs()
    mae = diff.max().item()
    denom = (a.abs().max() + 1e-8)
    mre = (diff / denom).max().item()
    return mae, mre


def test_step():
    device = 'mps'
    B, NH, DHQK, DHHV = 1, 2, 8, 8
    torch.manual_seed(0)
    # Inputs
    q = torch.randn(B, NH, DHQK, device=device, dtype=torch.float32)
    k = torch.randn(B, NH, DHQK, device=device, dtype=torch.float32)
    v = torch.randn(B, NH, DHHV, device=device, dtype=torch.float32)
    i = torch.randn(B, NH, 1, device=device, dtype=torch.float32)
    f = torch.randn(B, NH, 1, device=device, dtype=torch.float32)
    C0 = torch.zeros(B, NH, DHQK, DHHV, device=device, dtype=torch.float32)
    N0 = torch.zeros(B, NH, DHQK, device=device, dtype=torch.float32)
    M0 = torch.zeros(B, NH, 1, device=device, dtype=torch.float32)

    h_nat, (C_nat, N_nat, M_nat) = mlstm_recurrent_step__native_fw(matC_old=C0, vecN_old=N0, scaM_old=M0, vecQ=q,
                                                                   vecK=k,
                                                                   vecV=v, scaI=i, scaF=f)
    h_mtl, (C_mtl, N_mtl, M_mtl) = mlstm_recurrent_step__metal_fw(
        matC_old=C0, vecN_old=N0, scaM_old=M0.squeeze(-1), vecQ=q, vecK=k, vecV=v, scaI=i, scaF=f, eps=1e-6
    )
    # Harmonize M shape
    if M_mtl.ndim == 2:
        M_mtl = M_mtl.unsqueeze(-1)

    print("Step parity errors:")
    for name, a, b in [
        ("h", h_nat, h_mtl),
        ("C", C_nat, C_mtl),
        ("N", N_nat, N_mtl),
        ("M", M_nat, M_mtl),
    ]:
        mae, mre = max_err(a, b)
        print(f"  {name}: max_abs={mae:.3e} max_rel={mre:.3e}")

    # Alt Python metal formula to pinpoint differences
    DHQK = q.shape[-1]
    inv_sqrt = (DHQK ** -0.5)
    q_scaled = q * inv_sqrt
    f_log = torch.nn.functional.logsigmoid(f)
    m_new = torch.maximum(f_log + M0, i)
    f_act = torch.exp(f_log + M0 - m_new)
    i_act = torch.exp(i - m_new)
    C_alt = f_act[:, :, :, None] * C0 + i_act[:, :, :, None] * (k[:, :, :, None] * v[:, :, None, :])
    N_alt = f_act * N0 + i_act * k
    num_alt = (q_scaled[:, :, None, :] @ C_alt).squeeze(2)
    qn = (q_scaled[:, :, None, :] @ N_alt[:, :, :, None]).squeeze(2)
    denom_alt = torch.maximum(qn.abs(), torch.exp(-m_new)) + 1e-6
    h_alt = num_alt / denom_alt
    print("Alt-metal vs native:")
    for name, a, b in [("h", h_nat, h_alt), ("C", C_nat, C_alt), ("N", N_nat, N_alt), ("M", M_nat, m_new)]:
        mae, mre = max_err(a, b)
        print(f"  {name}: max_abs={mae:.3e} max_rel={mre:.3e}")


def test_sequence():
    device = 'mps'
    B, NH, S, DHQK, DHHV = 1, 2, 6, 8, 8
    torch.manual_seed(0)
    q = torch.randn(B, NH, S, DHQK, device=device, dtype=torch.float32)
    k = torch.randn(B, NH, S, DHQK, device=device, dtype=torch.float32)
    v = torch.randn(B, NH, S, DHHV, device=device, dtype=torch.float32)
    i = torch.randn(B, NH, S, device=device, dtype=torch.float32)
    f = torch.randn(B, NH, S, device=device, dtype=torch.float32)

    h_nat, _ = mlstm_recurrent_sequence__native_fw(q=q, k=k, v=v, i=i, f=f, return_last_states=True)
    # native_sequence__metal is registered via registry; fetch and call
    seq_metal = _seq_reg["native_sequence__metal"]
    h_mtl, _ = seq_metal(q=q, k=k, v=v, i=i, f=f, return_last_states=True, eps=1e-6)

    mae, mre = max_err(h_nat, h_mtl)
    print("Sequence parity errors:")
    print(f"  H: max_abs={mae:.3e} max_rel={mre:.3e}")


def main():
    """

    """
    assert torch.backends.mps.is_available(), "MPS required"
    test_step()
    test_sequence()


if __name__ == '__main__':
    main()
