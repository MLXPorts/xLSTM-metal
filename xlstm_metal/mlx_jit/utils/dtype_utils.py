"""Dtype Utilities – MLX Implementation (String to Dtype Mapping)

Overview
--------
Provides utilities for mapping human-readable dtype strings (from config files)
to MLX dtype objects. This enables configuration-driven precision selection
without hardcoding dtype objects.

Why String-Based Dtype Config?
-------------------------------
Configuration files (JSON, YAML) cannot directly encode programming objects.
By using string identifiers like "float32", "bfloat16", configs remain:
  - Human-readable and editable
  - Framework-agnostic (same strings work for MLX, PyTorch, JAX)
  - Version-stable (no serialization dependencies)

Supported Dtypes
----------------
- "float32" → mx.float32 (standard 32-bit floating point)
- "float16" → mx.float16 (half precision, 16-bit)
- "bfloat16" or "bf16" → mx.bfloat16 (brain float, 16-bit with float32 range)
- "fp16" → mx.float16 (alias for float16)

Mixed Precision Patterns
------------------------
Common configurations in xLSTM:
  - compute_dtype: "float32" or "bfloat16" (forward pass activations)
  - state_dtype: "float32" (recurrent state for precision)
  - param_dtype: "float32" (trainable parameters)

Using bfloat16 for compute while keeping state in float32 reduces memory
while preserving numerical stability in long-range accumulations.

BFloat16 vs Float16
-------------------
- **Float16**: Standard IEEE half precision
  - Exponent: 5 bits, Mantissa: 10 bits
  - Range: ~6e-8 to 65504
  - Issues: Limited range can cause overflow/underflow

- **BFloat16**: Brain float (Google TPU format)
  - Exponent: 8 bits (same as float32), Mantissa: 7 bits
  - Range: ~1e-38 to 3e38 (same as float32)
  - Advantages: Wider range, simpler float32 ↔ bf16 conversion

For LLM training, bfloat16 is typically preferred over float16 due to
better numerical stability with large models.

Usage
-----
In config.json:
  {
    "autocast_kernel_dtype": "bfloat16",
    "inference_state_dtype": "float32"
  }

In Python:
  from xlstm_metal.mlx_jit.utils import resolve_dtype
  compute_dtype = resolve_dtype(config['autocast_kernel_dtype'])
  state_dtype = resolve_dtype(config['inference_state_dtype'])

Fallback Behavior
-----------------
If an unknown string is provided, `resolve_dtype` falls back to the
default dtype (float32) to prevent crashes. This enables forward
compatibility with new dtype strings.

Parity
------
Logic mirrors torch-native dtype_utils for cross-backend compatibility.
"""

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
    """Map config dtype string to MLX dtype object.

    Parameters
    ----------
    name : str | None
        Dtype string from config ("float32", "bfloat16", "bf16", etc.).
        If None, uses default.
    default : str, default "float32"
        Fallback dtype string if name is None or unrecognized.

    Returns
    -------
    dtype : mx.Dtype
        Corresponding MLX dtype object.

    Examples
    --------
    >>> resolve_dtype("bfloat16")
    mlx.core.bfloat16
    >>> resolve_dtype(None)  # uses default
    mlx.core.float32
    >>> resolve_dtype("unknown_dtype")  # fallback to default
    mlx.core.float32
    """

    key = (name or default).lower()
    if key not in _DTYPE_MAP:
        key = default
    return _DTYPE_MAP[key]
