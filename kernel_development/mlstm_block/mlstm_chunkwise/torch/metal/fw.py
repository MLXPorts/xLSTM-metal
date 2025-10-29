#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Metal implementation of mLSTM forward pass (stub).

Currently delegates to the native autograd kernel while the Metal kernels
are being implemented. Keeps the registry stable so configs can select
"chunkwise--metal_autograd" safely on MPS.
"""

from pathlib import Path

import torch

_METAL_BACKEND = None

def _load_metal_backend() -> None:
    """Try to JIT-load the Metal backend extension if available.

    Does not raise on failure; leaves _METAL_BACKEND as None.
    """
    global _METAL_BACKEND
    if _METAL_BACKEND is not None:
        return
    try:
        from torch.utils.cpp_extension import load
        # Resolve repository root heuristically and locate the ObjC++ source
        here = Path(__file__).resolve()
        root = here
        mm_path = None
        for _ in range(6):
            # New preferred location
            candidate = (root / "../../../.." / "kernels/metal/pytorch_ext/mlstm_metal_backend.mm").resolve()
            if candidate.exists():
                mm_path = candidate
                break
            # Legacy location
            candidate2 = (root / "../../../.." / "mlstm_metal_kernels/mlstm_metal_backend.mm").resolve()
            if candidate2.exists():
                mm_path = candidate2
                break
            # Archived prototypes
            candidate3 = (root / "../../../.." / "research_archive/metal_prototypes/kernels_metal/pytorch_ext/mlstm_metal_backend.mm").resolve()
            if candidate3.exists():
                mm_path = candidate3
                break
            candidate4 = (root / "../../../.." / "research_archive/metal_prototypes/mlstm_metal_kernels/mlstm_metal_backend.mm").resolve()
            if candidate4.exists():
                mm_path = candidate4
                break
            root = root.parent
        if mm_path is None:
            return
        _METAL_BACKEND = load(
            name="mlstm_metal_backend",
            sources=[str(mm_path)],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
    except Exception:
        _METAL_BACKEND = None


def mlstm_chunkwise__metal_fw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    f: torch.Tensor,
    i: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    chunk_size: int = 64,
    return_last_states: bool = True,
    **kwargs
):
    """Metal implementation of mLSTM forward pass.

    Strict mode: no fallbacks. Raises if Metal backend or kernels are not implemented.
    """
    if not torch.backends.mps.is_available() or q.device.type != "mps":
        raise RuntimeError("Metal backend requires MPS device for inference")
    _load_metal_backend()
    if _METAL_BACKEND is None:
        raise RuntimeError("Failed to load Metal backend JIT (mlstm_metal_backend)")

    # Implement chunkwise forward by scanning sequence on GPU using the Metal step kernel.
    # Shapes: q,k,v: (B, NH, S, DH), f,i: (B, NH, S)
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]
    device = q.device

    # Initialize states
    if c_initial is None:
        c_state = torch.zeros((B, NH, DHQK, DHHV), dtype=torch.float32, device=device)
        n_state = torch.zeros((B, NH, DHQK), dtype=torch.float32, device=device)
        m_state = torch.zeros((B, NH), dtype=torch.float32, device=device)
    else:
        c_state = c_initial.to(torch.float32, device=device)
        n_state = n_initial.to(torch.float32, device=device)
        m_state = m_initial.to(torch.float32, device=device)

    H = []
    # Read shader source once
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
        raise FileNotFoundError("mlstm_kernels.metal not found for Metal chunkwise kernel")

    for t in range(S):
        q_t = q[:, :, t, :].to(torch.float32, device=device).contiguous()
        k_t = k[:, :, t, :].to(torch.float32, device=device).contiguous()
        v_t = v[:, :, t, :].to(torch.float32, device=device).contiguous()
        i_t = i[:, :, t].to(torch.float32, device=device).contiguous()
        f_t = f[:, :, t].to(torch.float32, device=device).contiguous()

        h_t, (c_state, n_state, m_state) = _METAL_BACKEND.metal_mlstm_step_with_source(
            q_t, k_t, v_t, i_t, f_t, c_state, n_state, m_state, float(kwargs.get("eps", 1e-6)), src
        )
        # h_t shape (B, NH, DHHV)
        H.append(h_t)

    matH = torch.stack(H, dim=-2)  # (B, NH, S, DHHV)
    if return_last_states:
        return matH, (c_state, n_state, m_state)
    return matH
