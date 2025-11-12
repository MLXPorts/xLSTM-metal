"""Utility helpers for mapping config dtype strings to MLX dtypes."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

_DTYPE_MAP = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
    "fp16": mx.float16,
}


def resolve_dtype(name: Optional[str], default: str = "float32") -> mx.Dtype:
    """Return an MLX dtype for the given config string."""

    key = (name or default).lower()
    if key not in _DTYPE_MAP:
        key = default
    return _DTYPE_MAP[key]
