"""Utility helpers for mapping config dtype strings to torch dtypes."""

from __future__ import annotations

from typing import Optional

import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def resolve_dtype(name: Optional[str], default: str = "float32") -> torch.dtype:
    """Return a torch dtype for the given config string.

    Falls back to the provided default if the name is unknown.
    """
    key = (name or default).lower()
    if key not in _DTYPE_MAP:
        key = default
    return _DTYPE_MAP[key]

__all__ = ["resolve_dtype"]
