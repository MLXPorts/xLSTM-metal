"""Safetensors shard loader for PyTorch backend."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open


def load_safetensor_shards(model_dir: str, index_filename: str = "model.safetensors.index.json") -> Dict[str, torch.Tensor]:
    """Load all safetensor shards listed in the index into torch tensors.

    Args:
        model_dir: Directory containing index + shard files
        index_filename: Name of index file

    Returns:
        Dict mapping weight_name -> torch.Tensor (on CPU)
    """
    md = Path(model_dir)
    index_path = md / index_filename
    if not index_path.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_path}")

    import json
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    if not shard_files:
        return {}

    weights: Dict[str, torch.Tensor] = {}

    for shard in shard_files:
        shard_path = md / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file missing: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as fshard:
            for name in fshard.keys():
                weights[name] = fshard.get_tensor(name)

    return weights

__all__ = ["load_safetensor_shards"]

