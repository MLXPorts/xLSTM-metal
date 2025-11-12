#!/usr/bin/env python
"""Legacy PyTorch NPZ weight loader (stub for torch backend).

The torch backend expects safetensors via `load_safetensor_shards`.
This module remains for backward compatibility but does not perform loading.
"""

from __future__ import annotations


def load_npz_weights_to_block(*_, **__):  # pragma: no cover
    raise RuntimeError("NPZ weight loading (PyTorch) not supported in torch_native backend.")


def load_weights_into_wired_model(*_, **__):  # pragma: no cover
    raise RuntimeError("NPZ weight loading into wired model not supported in torch_native backend.")


__all__ = ["load_npz_weights_to_block", "load_weights_into_wired_model"]
