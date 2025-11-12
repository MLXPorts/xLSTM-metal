#!/usr/bin/env python
"""Legacy PyTorch safetensors loader (deprecated for torch backend).

Replaced by `safetensors_torch_loader.load_safetensor_shards`.
This file now provides a thin compatibility shim that raises when used.
"""

def load_safetensors_into_wired_model(model_dir: str, model):  # pragma: no cover
    raise RuntimeError(
        "PyTorch-based safetensors loader is deprecated in torch_native. Use load_safetensor_shards instead."
    )

__all__ = ["load_safetensors_into_wired_model"]
