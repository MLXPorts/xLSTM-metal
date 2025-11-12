#!/usr/bin/env python
"""Infer model configuration from safetensors (PyTorch backend)."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from safetensors import safe_open


def _shape_of(tensor_name: str, index: Dict, shard_shapes: Dict[str, Dict[str, Tuple[int, ...]]]) -> Optional[Tuple[int, ...]]:
    if tensor_name in index["weight_map"]:
        shard_file = index["weight_map"][tensor_name]
        if shard_file in shard_shapes and tensor_name in shard_shapes[shard_file]:
            return shard_shapes[shard_file][tensor_name]
    # Fallback: metadata
    if "metadata" in index and tensor_name in index.get("metadata", {}):
        shape = index["metadata"][tensor_name].get("shape")
        if shape:
            return tuple(shape)
    return None


def _infer_heads_from_mhln(model_dir: str, d_model: int) -> int:
    # TODO: optionally inspect multihead_norm weight; for now assume 8 heads for 7B variant
    return 8


def infer_config_from_safetensors(model_dir: str) -> Dict[str, Any]:
    p = Path(model_dir)
    index_path = p / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_path}")
    with open(index_path) as f:
        index = json.load(f)

    # Collect shapes from first shard only for speed
    shard_files = sorted(set(index["weight_map"].values()))
    shard_shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    if shard_files:
        first = shard_files[0]
        shard_path = p / first
        if shard_path.exists():
            with safe_open(str(shard_path), framework="pt", device="cpu") as s:
                shard_shapes[first] = {name: tuple(s.get_tensor(name).shape) for name in s.keys()}

    def get_shape(name: str) -> Tuple[int, ...]:
        shape = _shape_of(name, index, shard_shapes)
        if shape is None:
            raise ValueError(f"Required tensor not found: {name}")
        return shape

    emb_shape = get_shape("backbone.embeddings.weight")
    lm_head_shape = get_shape("lm_head.weight")
    vocab_size, d_model = emb_shape
    if lm_head_shape != (vocab_size, d_model):
        raise ValueError("LM head shape mismatch with embeddings")

    block_ids = {int(k.split('.')[2]) for k in index['weight_map'].keys() if k.startswith('backbone.blocks.') and k.split('.')[2].isdigit()}
    if not block_ids:
        raise ValueError("No blocks found in checkpoint")
    num_blocks = max(block_ids) + 1

    block_0_prefix = "backbone.blocks.0"
    q_shape = get_shape(f"{block_0_prefix}.mlstm_layer.q.weight")
    v_shape = get_shape(f"{block_0_prefix}.mlstm_layer.v.weight")
    ig_b_shape = get_shape(f"{block_0_prefix}.mlstm_layer.igate_preact.bias")
    fg_b_shape = get_shape(f"{block_0_prefix}.mlstm_layer.fgate_preact.bias")
    out_proj_shape = get_shape(f"{block_0_prefix}.mlstm_layer.out_proj.weight")
    up_shape = get_shape(f"{block_0_prefix}.ffn.proj_up.weight")
    down_shape = get_shape(f"{block_0_prefix}.ffn.proj_down.weight")

    qk_dim, _ = q_shape
    ffn_hidden = up_shape[0]
    if down_shape[1] != ffn_hidden or down_shape[0] != d_model:
        raise ValueError("FFN shape inconsistency")
    if out_proj_shape != (d_model, d_model):
        raise ValueError("Output projection shape mismatch")

    qk_dim_factor = qk_dim / d_model
    ffn_proj_factor = ffn_hidden / d_model

    return {
        "embedding_dim": d_model,
        "vocab_size": vocab_size,
        "num_blocks": num_blocks,
        "qk_dim_factor": float(qk_dim_factor),
        "v_dim_factor": 1.0,
        "ffn_proj_factor": float(ffn_proj_factor),
        "num_heads": _infer_heads_from_mhln(model_dir, d_model),
        "gate_soft_cap": 15.0,
        "norm_eps": 1e-6,
        "use_bias": ig_b_shape is not None and fg_b_shape is not None,
        "output_logit_soft_cap": 30.0,
        "chunk_size": 64,
    }

__all__ = ["infer_config_from_safetensors"]
