import os
from typing import Tuple

import torch
import torch.nn.functional as F


def _stable_logsigmoid(x: torch.Tensor) -> torch.Tensor:
    # Equivalent to -softplus(-x) with better numerical stability
    return torch.where(x >= 0, -F.softplus(-x), x - F.softplus(x))


def _mlstm_step_eager(
        matC_old: torch.Tensor,  # (B, NH, DHQK, DHHV)
        vecN_old: torch.Tensor,  # (B, NH, DHQK)
        scaM_old: torch.Tensor,  # (B, NH, 1)
        vecQ: torch.Tensor,  # (B, NH, DHQK)
        vecK: torch.Tensor,  # (B, NH, DHQK)
        vecV: torch.Tensor,  # (B, NH, DHHV)
        scaI: torch.Tensor,  # (B, NH, 1)
        scaF: torch.Tensor,  # (B, NH, 1)
        eps: float = 1e-6,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Enforce float32 math like Triton kernels
    matC_old = matC_old.to(torch.float32)
    vecN_old = vecN_old.to(torch.float32)
    scaM_old = scaM_old.to(torch.float32)
    vecQ = vecQ.to(torch.float32)
    vecK = vecK.to(torch.float32)
    vecV = vecV.to(torch.float32)
    scaI = scaI.to(torch.float32)
    scaF = scaF.to(torch.float32)

    B, NH, DHQK = vecQ.shape
    DHHV = vecV.shape[-1]

    # Ensure shapes
    scaI = scaI.reshape(B, NH, 1)
    scaF = scaF.reshape(B, NH, 1)
    scaM_old = scaM_old.reshape(B, NH, 1)

    # Gating (per (B,NH,1))
    f_log = _stable_logsigmoid(scaF)
    m_new = torch.maximum(f_log + scaM_old, scaI)
    i_act = torch.exp(scaI - m_new)
    f_act = torch.exp(f_log + scaM_old - m_new)

    # q scaling
    q_scaled = vecQ * (DHQK ** -0.5)

    # C update: f*C_old + i*(k ⊗ v)  -> (B,NH,DHQK,DHHV)
    kv_outer = torch.einsum('bhj,bhd->bhjd', vecK, vecV)
    f_gate4 = f_act.unsqueeze(-1).expand_as(matC_old)
    i_gate4 = i_act.unsqueeze(-1).expand_as(matC_old)
    C_new = f_gate4 * matC_old + i_gate4 * kv_outer

    # N update: f*N_old + i*k  -> (B,NH,DHQK)
    N_new = f_act * vecN_old + i_act * vecK

    # Numerator and q·N
    h_num = torch.einsum('bhjd,bhj->bhd', C_new, q_scaled)  # (B,NH,DHHV)
    qn_dot = torch.einsum('bhj,bhj->bh', q_scaled, N_new)  # (B,NH)

    denom = torch.maximum(qn_dot.abs(), torch.exp(-m_new.squeeze(-1))).unsqueeze(-1) + eps
    H_new = h_num / denom

    return H_new, (C_new, N_new, m_new)


try:
    _mode = os.environ.get("XLSTM_COMPILE_MODE", "reduce-overhead")
    _mlstm_step_compiled_fn = torch.compile(_mlstm_step_eager, mode=_mode)
except Exception as e:
    raise RuntimeError(
        f"torch.compile failed for mLSTM metal step on this platform: {e}. "
        "This backend does not support fallbacks."
    )


def mlstm_recurrent_step__metal_fw(
        matC_old: torch.Tensor,  # (B, NH, DHQK, DHHV)
        vecN_old: torch.Tensor,  # (B, NH, DHQK)
        scaM_old: torch.Tensor,  # (B, NH, 1)
        vecQ: torch.Tensor,  # (B, NH, DHQK)
        vecK: torch.Tensor,  # (B, NH, DHQK)
        vecV: torch.Tensor,  # (B, NH, DHHV)
        scaI: torch.Tensor,  # (B, NH, 1)
        scaF: torch.Tensor,  # (B, NH, 1)
        eps: float = 1e-6,
        dtype_state: torch.dtype = torch.float32,
):
    """

    :param matC_old:
    :param vecN_old:
    :param scaM_old:
    :param vecQ:
    :param vecK:
    :param vecV:
    :param scaI:
    :param scaF:
    :param eps:
    :param dtype_state:
    :return:
    """
    # Strict device check: Metal backend must run on MPS
    dev = vecQ.device if 'vecQ' in locals() else q.device  # defensive
    if dev.type != 'mps':
        raise RuntimeError("mLSTM 'metal' step requires MPS device; CPU/CUDA not allowed.")
    return _mlstm_step_compiled_fn(matC_old, vecN_old, scaM_old, vecQ, vecK, vecV, scaI, scaF, eps)


def mlstm_recurrent_step__metal(
        q: torch.Tensor,  # (B, NH, DHQK)
        k: torch.Tensor,  # (B, NH, DHQK)
        v: torch.Tensor,  # (B, NH, DHV)
        i: torch.Tensor,  # (B, NH, 1)
        f: torch.Tensor,  # (B, NH, 1)
        c: torch.Tensor,  # (B, NH, DHQK, DHV)
        n: torch.Tensor,  # (B, NH, DHQK)
        m: torch.Tensor,  # (B, NH, 1)
        eps: float = 1e-6,
        dtype_state: torch.dtype = torch.float32,
):
    """

    :param q:
    :param k:
    :param v:
    :param i:
    :param f:
    :param c:
    :param n:
    :param m:
    :param eps:
    :param dtype_state:
    :return:
    """
    # Strict device check: Metal backend must run on MPS
    if q.device.type != 'mps':
        raise RuntimeError("mLSTM 'metal' step requires MPS device; CPU/CUDA not allowed.")
    return _mlstm_step_compiled_fn(c, n, m, q, k, v, i, f, eps)
