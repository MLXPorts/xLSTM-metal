#!/usr/bin/env python
"""
Direct safetensors weight loader for xLSTM-7B MAD model.

Loads HuggingFace safetensors directly into WiredMADModel without conversion.
Uses the canonical match_dict from xlstm-jax.
"""

import mlx.core as mx
import json
from pathlib import Path
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..wiring import WiredMADModel


def load_safetensors_into_wired_model(model_dir: str, model: "WiredMADModel"):
    """
    Load xLSTM-7B weights directly from HuggingFace safetensors into WiredMADModel.

    Based on xlstm-jax canonical implementation:
    xlstm_jax/utils/model_param_handling/handle_mlstm_simple.py

    Args:
        model_dir: Path to HuggingFace model directory with safetensors
        model: WiredMADModel instance to load weights into
    """
    model_dir = Path(model_dir)

    print(f"Loading weights from HuggingFace safetensors: {model_dir}")

    # Load index
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Load all shards
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Loading {len(shard_files)} shards...")

    hf_weights = {}
    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        shard_weights = mx.load(str(shard_path))
        hf_weights.update(shard_weights)

    print(f"Loaded {len(hf_weights)} tensors from safetensors")

    # Load embedding
    if 'backbone.embeddings.weight' in hf_weights and 'embedding' in model.blocks:
        model.blocks['embedding'].weight = hf_weights['backbone.embeddings.weight']
        print(f"✓ Loaded embedding: {hf_weights['backbone.embeddings.weight'].shape}")

    # Load blocks
    num_blocks = sum(1 for name in model.blocks.keys() if name.startswith('xlstm_'))
    print(f"Loading {num_blocks} xLSTM blocks...")

    for i in range(num_blocks):
        block_name = f'xlstm_{i}'
        if block_name not in model.blocks:
            print(f"Warning: {block_name} not found in model")
            continue

        block = model.blocks[block_name]
        hf_prefix = f'backbone.blocks.{i}'

        # mLSTM layer weights
        mlstm_mapping = {
            f'{hf_prefix}.mlstm_layer.q.weight': ('xlstm', 'q', 'weight'),
            f'{hf_prefix}.mlstm_layer.k.weight': ('xlstm', 'k', 'weight'),
            f'{hf_prefix}.mlstm_layer.v.weight': ('xlstm', 'v', 'weight'),
            f'{hf_prefix}.mlstm_layer.igate_preact.weight': ('xlstm', 'igate_preact', 'weight'),
            f'{hf_prefix}.mlstm_layer.igate_preact.bias': ('xlstm', 'igate_preact', 'bias'),
            f'{hf_prefix}.mlstm_layer.fgate_preact.weight': ('xlstm', 'fgate_preact', 'weight'),
            f'{hf_prefix}.mlstm_layer.fgate_preact.bias': ('xlstm', 'fgate_preact', 'bias'),
            f'{hf_prefix}.mlstm_layer.ogate_preact.weight': ('xlstm', 'ogate_preact', 'weight'),
            f'{hf_prefix}.mlstm_layer.out_proj.weight': ('xlstm', 'out_proj', 'weight'),
        }

        for hf_key, (module, submodule, param) in mlstm_mapping.items():
            if hf_key in hf_weights:
                target = getattr(getattr(block, module), submodule)
                setattr(target, param, hf_weights[hf_key])

        # Multi-head layer norm - needs reshaping
        mhln_key = f'{hf_prefix}.mlstm_layer.multihead_norm.weight'
        if mhln_key in hf_weights:
            mhln_weight = hf_weights[mhln_key]
            num_heads = block.xlstm.config.num_heads
            head_dim = block.xlstm.config.head_dim
            block.xlstm.multihead_norm.weight = mhln_weight.reshape(num_heads, head_dim)

        # Norms
        norm_mlstm_key = f'{hf_prefix}.norm_mlstm.weight'
        if norm_mlstm_key in hf_weights:
            block.xlstm_norm.weight = hf_weights[norm_mlstm_key]

        norm_ffn_key = f'{hf_prefix}.norm_ffn.weight'
        if norm_ffn_key in hf_weights:
            block.ffn_norm.weight = hf_weights[norm_ffn_key]

        # FFN - concatenate proj_up_gate and proj_up
        up_gate_key = f'{hf_prefix}.ffn.proj_up_gate.weight'
        up_key = f'{hf_prefix}.ffn.proj_up.weight'
        if up_gate_key in hf_weights and up_key in hf_weights:
            # Concatenate gate and up projections
            up_gate = hf_weights[up_gate_key]  # [10944, 4096]
            up = hf_weights[up_key]  # [10944, 4096]
            block.ffn.proj_up.weight = mx.concatenate([up_gate, up], axis=0)  # [21888, 4096]

        down_key = f'{hf_prefix}.ffn.proj_down.weight'
        if down_key in hf_weights:
            block.ffn.proj_down.weight = hf_weights[down_key]

        if (i + 1) % 8 == 0:
            print(f"  ✓ Loaded blocks 0-{i}")

    # Load output norm
    if 'backbone.out_norm.weight' in hf_weights and 'out_norm' in model.blocks:
        model.blocks['out_norm'].weight = hf_weights['backbone.out_norm.weight']
        print(f"✓ Loaded out_norm: {hf_weights['backbone.out_norm.weight'].shape}")

    # Load LM head
    if 'lm_head.weight' in hf_weights and 'lm_head' in model.blocks:
        model.blocks['lm_head'].weight = hf_weights['lm_head.weight']
        print(f"✓ Loaded lm_head: {hf_weights['lm_head.weight'].shape}")

    print(f"\n✅ Successfully loaded all pretrained weights from safetensors")
