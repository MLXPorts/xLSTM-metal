"""Metal causal conv loaders for PyTorch (MPS)."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "xlstm_causal_conv1d"
_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend is required for causal conv Metal kernels")
    src = Path(__file__).with_name("causal_conv_extension.mm")
    _EXT = load(
        name=_EXT_NAME,
        sources=[str(src)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )
    return _EXT


def causal_conv1d_mixing(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    ext = _load_ext()
    return ext.mixing_forward(x, weight, bias)


def causal_conv1d_depthwise(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    ext = _load_ext()
    return ext.depthwise_forward(x, weight, bias)
