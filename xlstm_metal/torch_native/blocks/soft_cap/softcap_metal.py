"""Soft-cap Metal kernel loader (PyTorch MPS)."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "xlstm_soft_cap"
_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend is required for soft-cap Metal kernel")
    src_path = Path(__file__).with_name("soft_cap_extension.mm")
    _EXT = load(
        name=_EXT_NAME,
        sources=[str(src_path)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )
    return _EXT


def soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    ext = _load_ext()
    return ext.soft_cap_forward(x, float(cap))
