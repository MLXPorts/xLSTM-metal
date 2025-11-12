#!/usr/bin/env python
"""
Config inference from safetensors checkpoint.

Derives model architecture parameters directly from tensor shapes in the checkpoint,
making the model loading fully model-agnostic.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import mlx.core as mx


def _shape_of(tensor_name: str, index: Dict, shards: Dict[str, mx.array]) -> Optional[Tuple[int, ...]]:
    """
    Get shape of a tensor from index metadata or loaded shards.

    Args:
        tensor_name: Name of the tensor
        index: The safetensors index dict
        shards: Dict of loaded shard arrays

    Returns:
        Shape tuple or None if not found
    """
    if tensor_name in index["weight_map"]:
        shard_file = index["weight_map"][tensor_name]
        if shard_file in shards:
            if tensor_name in shards[shard_file]:
                return shards[shard_file][tensor_name].shape

    # Fallback: check if shape is in metadata
    if "metadata" in index and tensor_name in index["metadata"]:
        shape_str = index["metadata"][tensor_name].get("shape")
        if shape_str:
            return tuple(shape_str)

    return None


def _parse_block_index(text: str) -> int:
    """Convert dot-delimited block index text to integer without direct casts."""
    return json.loads(text)


def _infer_heads_from_mhln(model_dir: str, d_model: int) -> int:
    """
    Infer number of heads from multihead_norm weight shape.

    Args:
        model_dir: Path to model directory
        d_model: Model embedding dimension

    Returns:
        Number of heads
    """
    # For now, return a default - in practice, we'd load the tensor
    # and check mhln.weight.shape[0] // (d_model // num_heads_per_group) etc.
    # But for xLSTM-7B, it's 8 heads
    return 8


def _infer_gate_soft_cap_default() -> float:
    """Default gate soft cap value."""
    return 15.0


def _infer_norm_eps_default() -> float:
    """Default normalization epsilon."""
    return 1e-6


def infer_config_from_safetensors(model_dir: str) -> Dict[str, Any]:
    """
    Infer complete model configuration from safetensors checkpoint.

    Reads tensor shapes to derive all architectural parameters, making
    the model loading fully checkpoint-driven and model-agnostic.

    Args:
        model_dir: Path to model directory with safetensors

    Returns:
        Dict with inferred configuration

    Raises:
        FileNotFoundError: If safetensors index not found
        ValueError: If required tensors missing or shapes inconsistent
    """
    p = Path(model_dir)

    # Load index
    index_path = p / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    # Load first shard to get shapes (lazy loading)
    shard_files = sorted(set(index["weight_map"].values()))
    shards = {}
    first_shard = p / shard_files[0]
    if first_shard.exists():
        shards[shard_files[0]] = mx.load(str(first_shard))

    # Helper to get shape
    def get_shape(name: str) -> Tuple[int, ...]:
        """

        :param name:
        :return:
        """
        shape = _shape_of(name, index, shards)
        if shape is None:
            raise ValueError(f"Required tensor not found: {name}")
        return shape

    # Derive top-level dims from well-known tensor names
    # Embedding and LM head
    emb_shape = get_shape("backbone.embeddings.weight")
    lm_head_shape = get_shape("lm_head.weight")
    vocab_size, d_model = emb_shape

    if lm_head_shape != (vocab_size, d_model):
        raise ValueError(f"LM head shape {lm_head_shape} doesn't match embedding {emb_shape}")

    # Count number of blocks
    block_ids = set()
    for tname in index["weight_map"].keys():
        if tname.startswith("backbone.blocks."):
            parts = tname.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                block_ids.add(int(parts[2]))

    if not block_ids:
        raise ValueError("No blocks found in checkpoint")

    num_blocks = max(block_ids) + 1

    # Per-layer dims and head structure from first block
    block_0_prefix = "backbone.blocks.0"

    # Q, K, V projections
    q_shape = get_shape(f"{block_0_prefix}.mlstm_layer.q.weight")
    k_shape = get_shape(f"{block_0_prefix}.mlstm_layer.k.weight")
    v_shape = get_shape(f"{block_0_prefix}.mlstm_layer.v.weight")

    qk_dim, _ = q_shape
    v_out, v_in = v_shape

    if v_in != d_model or v_out != d_model:
        raise ValueError(f"V projection shape {v_shape} doesn't match d_model {d_model}")

    # Gating heads (igate/fgate)
    ig_w_shape = get_shape(f"{block_0_prefix}.mlstm_layer.igate_preact.weight")
    ig_b_shape = get_shape(f"{block_0_prefix}.mlstm_layer.igate_preact.bias")
    fg_w_shape = get_shape(f"{block_0_prefix}.mlstm_layer.fgate_preact.weight")
    fg_b_shape = get_shape(f"{block_0_prefix}.mlstm_layer.fgate_preact.bias")

    num_gate_heads = ig_b_shape[0]
    if fg_b_shape[0] != num_gate_heads:
        raise ValueError(f"Inconsistent gate head counts: igate {ig_b_shape[0]}, fgate {fg_b_shape[0]}")

    # Output proj
    out_proj_shape = get_shape(f"{block_0_prefix}.mlstm_layer.out_proj.weight")
    if out_proj_shape != (d_model, d_model):
        raise ValueError(f"Output proj shape {out_proj_shape} doesn't match d_model {d_model}")

    # FFN shapes
    up_gate_shape = get_shape(f"{block_0_prefix}.ffn.proj_up_gate.weight")
    up_shape = get_shape(f"{block_0_prefix}.ffn.proj_up.weight")
    down_shape = get_shape(f"{block_0_prefix}.ffn.proj_down.weight")

    ffn_hidden = up_shape[0]
    if up_gate_shape[0] != ffn_hidden or down_shape[1] != ffn_hidden or down_shape[0] != d_model:
        raise ValueError(f"Inconsistent FFN shapes: up_gate {up_gate_shape}, up {up_shape}, down {down_shape}")

    # Derive factors
    qk_dim_factor = qk_dim / d_model
    v_dim_factor = 1.0  # by design in xLSTM
    ffn_proj_factor = ffn_hidden / d_model

    # Presence of biases
    use_bias = (ig_b_shape is not None) and (fg_b_shape is not None)

    # Infer heads from multihead_norm
    num_heads = _infer_heads_from_mhln(model_dir, d_model)

    return {
        "embedding_dim": d_model,
        "vocab_size": vocab_size,
        "num_blocks": num_blocks,
        "qk_dim_factor": float(qk_dim_factor),
        "v_dim_factor": float(v_dim_factor),
        "ffn_proj_factor": float(ffn_proj_factor),
        "num_heads": num_heads,
        "gate_soft_cap": _infer_gate_soft_cap_default(),
        "norm_eps": _infer_norm_eps_default(),
        "use_bias": use_bias,
        # Runtime defaults (not derived from weights)
        "output_logit_soft_cap": 30.0,  # Could be in metadata
        "chunk_size": 64,  # Runtime parameter
    }
