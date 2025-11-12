"""Torch-native wrapper for the mLSTM chunkwise recurrent Metal kernel.

This module mirrors the MLX implementation but drives a custom Metal shader
through PyTorch's MPS backend. The shader source lives in
`mlstm_chunkwise_recurrent_fw_C_kernel.metal` and is loaded via the Objective-C++
launcher in the same folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "xlstm_mlstm_chunkwise_recurrent_fw_C"
_EXT = None


def _load_ext():
    """Load (or re-use) the compiled Metal extension for the recurrent kernel."""
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend is required for the mLSTM recurrent kernel")

    src_path = Path(__file__).with_name("mlstm_chunkwise_recurrent_fw_C_extension.mm")
    _EXT = load(
        name=_EXT_NAME,
        sources=[str(src_path)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )
    return _EXT


def _ensure_tensor(
    tensor: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(shape, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype, copy=False)
    if tensor.shape != shape:
        raise ValueError(f"Expected shape {shape} but received {tensor.shape}")
    return tensor.contiguous()


def mlstm_chunkwise_recurrent_fw_C_metal(
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecF: torch.Tensor,
    vecI: torch.Tensor,
    matC_initial: Optional[torch.Tensor],
    vecN_initial: Optional[torch.Tensor],
    scaMinter_initial: Optional[torch.Tensor],
    NC: int,
    L: int,
    siz_b_DHQK: int = 16,
    siz_b_DHHV: int = 16,
    save_states_every_nth_chunk: int = 1,
    dbg: Optional[torch.Tensor] = None,
    state_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the chunkwise recurrent kernel via Metal on MPS."""

    if matK.device.type != "mps":
        raise ValueError("matK must reside on the MPS device")
    device = matK.device
    if matK.dtype != torch.float32:
        raise ValueError("matK must be float32 for the Metal kernel")

    matV = matV.to(device=device, dtype=torch.float32, copy=False)
    vecF = vecF.to(device=device, dtype=torch.float32, copy=False)
    vecI = vecI.to(device=device, dtype=torch.float32, copy=False)

    if not (matK.shape[:3] == vecF.shape == vecI.shape == matV.shape[:3]):
        raise ValueError("Input projections and gate tensors must share (B, NH, S)")

    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[3]

    states_dtype = state_dtype or torch.float32

    matC_states = torch.zeros((B, NH, (NC + 1) * DHQK, DHHV), device=device, dtype=states_dtype)
    vecN_states = torch.zeros((B, NH, (NC + 1) * DHQK), device=device, dtype=states_dtype)
    scaMinter_states = torch.zeros((B, NH, NC + 1), device=device, dtype=states_dtype)

    has_initial = matC_initial is not None
    matC_initial = _ensure_tensor(matC_initial, (B, NH, DHQK, DHHV), device, states_dtype)
    vecN_initial = _ensure_tensor(vecN_initial, (B, NH, DHQK), device, states_dtype)
    scaMinter_initial = _ensure_tensor(scaMinter_initial, (B, NH), device, states_dtype)

    dbg_requested = dbg is not None
    if dbg is None:
        dbg = torch.zeros((L * 3 + 1,), device=device, dtype=torch.float32)
    else:
        dbg = dbg.to(device=device, dtype=torch.float32, copy=False).contiguous()

    USE_INITIAL_STATE = 1 if has_initial else 0
    USE_DBG = 1 if dbg_requested else 0

    params = torch.tensor(
        [
            B,
            NH,
            S,
            DHQK,
            DHHV,
            NC,
            L,
            siz_b_DHQK,
            siz_b_DHHV,
            save_states_every_nth_chunk,
            USE_INITIAL_STATE,
            USE_DBG,
        ],
        dtype=torch.int32,
        device=device,
    )

    strides = torch.tensor(
        [
            NH * S * DHQK,
            DHQK,
            1,
            NH * S * DHHV,
            DHHV,
            1,
            NH * S,
            (NC + 1) * DHQK * DHHV,
            DHHV,
            1,
            (NC + 1) * DHQK,
            1,
            NC + 1,
            1,
            NH * DHQK * DHHV,
            DHHV,
            1,
            NH * DHQK,
            1,
            NH,
        ],
        dtype=torch.int32,
        device=device,
    )

    num_tiles_DHQK = (DHQK + siz_b_DHQK - 1) // siz_b_DHQK
    num_tiles_DHHV = (DHHV + siz_b_DHHV - 1) // siz_b_DHHV
    grid_x, grid_y, grid_z = num_tiles_DHQK, num_tiles_DHHV, B * NH
    threadgroup_x, threadgroup_y = siz_b_DHHV, siz_b_DHQK

    ext = _load_ext()
    ext.mlstm_recurrent_fw_C_forward(
        matK.contiguous(),
        matV.contiguous(),
        vecF.contiguous(),
        vecI.contiguous(),
        matC_initial,
        vecN_initial,
        scaMinter_initial,
        matC_states,
        vecN_states,
        scaMinter_states,
        dbg,
        params,
        strides,
        int(grid_x),
        int(grid_y),
        int(grid_z),
        int(threadgroup_x),
        int(threadgroup_y),
    )

    if dbg_requested:
        return matC_states, vecN_states, scaMinter_states, dbg
    return matC_states, vecN_states, scaMinter_states


__all__ = ["mlstm_chunkwise_recurrent_fw_C_metal"]
