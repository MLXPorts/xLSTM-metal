from __future__ import annotations

from pathlib import Path

import torch

_BACKEND = None

def _find_backend_source(max_up: int = 6) -> Path | None:
    here = Path(__file__).resolve()
    root = here
    for _ in range(max_up):
        candidate = (root / "../../../.." / "kernels/metal/pytorch_ext/mlstm_metal_backend.mm").resolve()
        if candidate.exists():
            return candidate
        legacy = (root / "../../../.." / "mlstm_metal_kernels/mlstm_metal_backend.mm").resolve()
        if legacy.exists():
            return legacy
        arc = (root / "../../../.." / "research_archive/metal_prototypes/kernels_metal/pytorch_ext/mlstm_metal_backend.mm").resolve()
        if arc.exists():
            return arc
        arc2 = (root / "../../../.." / "research_archive/metal_prototypes/mlstm_metal_kernels/mlstm_metal_backend.mm").resolve()
        if arc2.exists():
            return arc2
        root = root.parent
    return None

def _load_backend() -> None:
    global _BACKEND
    if _BACKEND is not None:
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available; Metal backend required")
    mm = _find_backend_source()
    if mm is None:
        raise ImportError("Metal backend source not found (kernels/metal/pytorch_ext/mlstm_metal_backend.mm)")
    from torch.utils.cpp_extension import load
    _BACKEND = load(
        name="mlstm_metal_backend",
        sources=[str(mm)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )

def mlstm_recurrent_step__metal_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH)
    vecQ: torch.Tensor,      # (B, NH, DHQK)
    vecK: torch.Tensor,      # (B, NH, DHQK)
    vecV: torch.Tensor,      # (B, NH, DHV)
    scaI: torch.Tensor,      # (B, NH, 1) or (B, NH)
    scaF: torch.Tensor,      # (B, NH, 1) or (B, NH)
    eps: float = 1e-6,
    dtype_state: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available; Metal backend required")
    device = vecQ.device
    if device.type != "mps":
        raise RuntimeError("All inputs must be on MPS device")

    _load_backend()

    # Ensure dtype and shapes
    # States are kept in float32 for numerical parity with Triton
    matC = matC_old.to(dtype=torch.float32, device=device).contiguous()
    vecN = vecN_old.to(dtype=torch.float32, device=device).contiguous()
    scaM = scaM_old.to(dtype=torch.float32, device=device).contiguous()

    vecQ32 = vecQ.to(dtype=torch.float32, device=device).contiguous()
    vecK32 = vecK.to(dtype=torch.float32, device=device).contiguous()
    vecV32 = vecV.to(dtype=torch.float32, device=device).contiguous()

    scaI = scaI.squeeze(-1) if scaI.dim() == 3 else scaI
    scaF = scaF.squeeze(-1) if scaF.dim() == 3 else scaF
    scaI32 = scaI.to(dtype=torch.float32, device=device).contiguous()
    scaF32 = scaF.to(dtype=torch.float32, device=device).contiguous()

    # Call Metal kernel
    # Read shader source (avoid duplication)
    here = Path(__file__).resolve()
    root = here
    src = None
    for _ in range(6):
        met = (root / "../../../.." / "kernels/metal/shaders/mlstm_kernels.metal").resolve()
        if met.exists():
            src = met.read_text()
            break
        legacy = (root / "../../../.." / "mlstm_metal_kernels/mlstm_kernels.metal").resolve()
        if legacy.exists():
            src = legacy.read_text()
            break
        arc = (root / "../../../.." / "research_archive/metal_prototypes/kernels_metal/shaders/mlstm_kernels.metal").resolve()
        if arc.exists():
            src = arc.read_text()
            break
        arc2 = (root / "../../../.." / "research_archive/metal_prototypes/mlstm_metal_kernels/mlstm_kernels.metal").resolve()
        if arc2.exists():
            src = arc2.read_text()
            break
        root = root.parent
    if src is None:
        raise FileNotFoundError("mlstm_kernels.metal not found for Metal step kernel")

    H, C_new, N_new, M_new = _BACKEND.metal_mlstm_step_with_source(
        vecQ32, vecK32, vecV32, scaI32, scaF32, matC, vecN, scaM, float(eps), src
    )

    # H has shape (B, NH, DHV)
    return H, (C_new, N_new, M_new)


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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Use the fw function with mapped arguments
    return mlstm_recurrent_step__metal_fw(
        matC_old=c,
        vecN_old=n,
        scaM_old=m,
        vecQ=q,
        vecK=k,
        vecV=v,
        scaI=i,
        scaF=f,
        eps=eps,
        dtype_state=dtype_state,
    )
