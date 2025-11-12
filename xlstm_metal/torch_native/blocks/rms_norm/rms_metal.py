"""RMSNorm Metal kernel loader (PyTorch MPS)."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "xlstm_rms_norm"
_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend is required for RMSNorm Metal kernel")
    src = Path(__file__).with_name("rms_norm_extension.mm")
    _EXT = load(
        name=_EXT_NAME,
        sources=[str(src)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )
    return _EXT


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ext = _load_ext()
    return ext.rms_norm_forward(x, weight, float(eps))
