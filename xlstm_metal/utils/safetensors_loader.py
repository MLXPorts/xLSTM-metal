#!/usr/bin/env python
"""
Simple safetensors weight loader for xLSTM-7B MAD model.

Just loads the dict and maps HuggingFace names to our block names.
No manual assignment - let the blocks handle their own parameter structure.
"""

import mlx.core as mx
import json
from pathlib import Path
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..wiring import WiredMADModel


def load_safetensors_into_wired_model(model_dir: str, model: "WiredMADModel"):
    """
    Load xLSTM-7B weights from HuggingFace safetensors.

    Simple approach:
    1. Load all safetensors into a dict (mx.load does this)
    2. Map HuggingFace keys to our block keys
    3. Assign weights to blocks

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

    # Load all shards into one dict
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Loading {len(shard_files)} shards...")

    hf_weights = {}
    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        shard_weights = mx.load(str(shard_path))
        hf_weights.update(shard_weights)

    print(f"Loaded {len(hf_weights)} tensors from safetensors")

    # Map HuggingFace keys to our block structure
    # HF: backbone.blocks.{i}.mlstm_layer.q.weight
    # Us: blocks['xlstm_{i}'].xlstm.q.weight

    # Build a dict that maps our parameter paths to HF parameter values
    our_weights = {}

    # Embedding
    if 'backbone.embeddings.weight' in hf_weights:
        our_weights['embedding.weight'] = hf_weights['backbone.embeddings.weight']

    # xLSTM blocks
    num_blocks = sum(1 for name in model.blocks.keys() if name.startswith('xlstm_'))
    for i in range(num_blocks):
        hf_prefix = f'backbone.blocks.{i}'
        our_prefix = f'xlstm_{i}'

        # Map all HF keys for this block to our keys
        for hf_key, tensor in hf_weights.items():
            if not hf_key.startswith(hf_prefix):
                continue

            # Strip HF prefix and map to our structure
            suffix = hf_key[len(hf_prefix)+1:]  # Remove "backbone.blocks.{i}."

            # Map component names
            suffix = suffix.replace('mlstm_layer.', 'xlstm.')
            suffix = suffix.replace('norm_mlstm.', 'xlstm_norm.')
            suffix = suffix.replace('norm_ffn.', 'ffn_norm.')
            # ffn. stays ffn.

            our_key = f'{our_prefix}.{suffix}'
            our_weights[our_key] = tensor

    # Output norm
    if 'backbone.out_norm.weight' in hf_weights:
        our_weights['out_norm.weight'] = hf_weights['backbone.out_norm.weight']

    # LM head
    if 'lm_head.weight' in hf_weights:
        our_weights['lm_head.weight'] = hf_weights['lm_head.weight']

    # Now assign weights to blocks by iterating through our_weights
    print(f"Mapping {len(our_weights)} weights to model blocks...")

    for our_key, tensor in our_weights.items():
        try:
            # Parse key: "xlstm_0.xlstm.q.weight" -> block="xlstm_0", path="xlstm.q.weight"
            parts = our_key.split('.', 1)
            if len(parts) != 2:
                # Top-level like "embedding.weight"
                block_name = parts[0]
                param_path = 'weight'
            else:
                block_name, param_path = parts

            if block_name not in model.blocks:
                print(f"Warning: Block '{block_name}' not in model, skipping {our_key}")
                continue

            # Navigate to the parameter and set it
            block = model.blocks[block_name]
            path_parts = param_path.split('.')

            # Navigate to the parent module
            obj = block
            for part in path_parts[:-1]:
                if not part:  # Skip empty parts
                    continue
                obj = getattr(obj, part)

            # Set the parameter
            param_name = path_parts[-1]
            if not param_name:
                print(f"Warning: Empty parameter name for {our_key}")
                continue

            setattr(obj, param_name, tensor)

        except Exception as e:
            print(f"Error loading {our_key}: {e}")
            raise

    print(f"\n✅ Successfully loaded all pretrained weights from safetensors")
