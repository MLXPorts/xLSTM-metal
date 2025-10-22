#!/usr/bin/env python
"""
Convert xLSTM-7B HuggingFace safetensors to MAD NPZ format.

Uses the EXACT match_dict from xlstm-jax canonical implementation:
/Volumes/stuff/Projects/xlstm-jax/xlstm_jax/utils/model_param_handling/handle_mlstm_simple.py
Lines 102-121
"""

import json
from pathlib import Path
import mlx.core as mx


def load_hf_safetensors(model_dir: str) -> dict:
    """
    Load all sharded safetensors files using MLX native loading.

    Args:
        model_dir: Directory containing model-*.safetensors files

    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    model_dir = Path(model_dir)

    # Load index to find shards
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Loading {len(shard_files)} shard files...")

    # Use MLX's native safetensors loading
    weights = {}
    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        print(f"  Loading {shard_file}...")
        shard_weights = mx.load(str(shard_path))
        weights.update(shard_weights)

    print(f"Loaded {len(weights)} tensors")
    return weights


def convert_to_mad_format(hf_weights: dict) -> dict:
    """
    Convert HuggingFace weight names to MAD NPZ format.

    Uses the EXACT match_dict from JAX canonical code:
    xlstm_jax/utils/model_param_handling/handle_mlstm_simple.py lines 102-121

    JAX match_dict (from_key -> to_key):
        "lm_head.out_dense.kernel": "lm_head.weight"
        "lm_head.out_norm.scale": "out_norm.weight"
        "embedding": "embedding.weight"
        ".ffn.proj_up.Dense_0.kernel": ".ffn.proj_up.weight"
        ".ffn.proj_down.Dense_0.kernel": ".ffn.proj_down.weight"
        ".ffn.proj_up_gate.Dense_0.kernel": ".ffn.proj_up_gate.weight"
        ".ffn_norm.sharded.scale": ".norm_ffn.weight"
        ".xlstm.dense_k.Dense_0.kernel": "mlstm_layer.k.weight"
        ".xlstm.dense_q.Dense_0.kernel": "mlstm_layer.q.weight"
        ".xlstm.dense_v.Dense_0.kernel": "mlstm_layer.v.weight"
        ".xlstm.dense_o.Dense_0.kernel": "mlstm_layer.ogate_preact.weight"
        ".xlstm.fgate.Dense_0.kernel": "mlstm_layer.fgate_preact.weight"
        ".xlstm.fgate.Dense_0.bias": "mlstm_layer.fgate_preact.bias"
        ".xlstm.igate.Dense_0.kernel": "mlstm_layer.igate_preact.weight"
        ".xlstm.igate.Dense_0.bias": "mlstm_layer.igate_preact.bias"
        ".xlstm.outnorm.sharded.scale": "mlstm_layer.multihead_norm.weight"
        ".xlstm.proj_down.Dense_0.kernel": "mlstm_layer.out_proj.weight"
        ".xlstm_norm.sharded.scale": ".norm_mlstm.weight"

    HF Format (backbone prefix):
        backbone.embeddings.weight
        backbone.blocks.{i}.mlstm_layer.*
        backbone.blocks.{i}.norm_mlstm.weight
        backbone.blocks.{i}.norm_ffn.weight
        backbone.blocks.{i}.ffn.*
        backbone.out_norm.weight
        lm_head.weight

    Weight Loader Format (what weight_loader.py expects):
        embedding.weight
        blocks.{i}.W_q.weight
        blocks.{i}.W_k.weight
        blocks.{i}.W_v.weight
        blocks.{i}.W_i.weight/bias
        blocks.{i}.W_f.weight/bias
        blocks.{i}.W_o.weight
        blocks.{i}.mhln.weight
        blocks.{i}.norm.weight
        blocks.{i}.norm2.weight
        blocks.{i}.up_l_proj.weight (proj_up_gate)
        blocks.{i}.up_r_proj.weight (proj_up)
        blocks.{i}.down_proj.weight
        out_norm.weight
        head.W
    """
    mad_weights = {}

    # Embedding
    if "backbone.embeddings.weight" in hf_weights:
        mad_weights["embedding.weight"] = hf_weights["backbone.embeddings.weight"]
        print("✓ Mapped embedding")

    # Detect number of blocks
    num_blocks = max([int(k.split(".")[2]) for k in hf_weights.keys() if k.startswith("backbone.blocks.")]) + 1
    print(f"Found {num_blocks} xLSTM blocks")

    # Convert each block using JAX canonical mapping
    for i in range(num_blocks):
        hf_prefix = f"backbone.blocks.{i}"
        mad_prefix = f"blocks.{i}"

        # mLSTM layer - mapping from HF to weight_loader format
        mlstm_mapping = {
            # HF format -> weight_loader format
            f"{hf_prefix}.mlstm_layer.q.weight": f"{mad_prefix}.W_q.weight",
            f"{hf_prefix}.mlstm_layer.k.weight": f"{mad_prefix}.W_k.weight",
            f"{hf_prefix}.mlstm_layer.v.weight": f"{mad_prefix}.W_v.weight",
            f"{hf_prefix}.mlstm_layer.out_proj.weight": f"{mad_prefix}.W_o.weight",
            f"{hf_prefix}.mlstm_layer.igate_preact.weight": f"{mad_prefix}.W_i.weight",
            f"{hf_prefix}.mlstm_layer.igate_preact.bias": f"{mad_prefix}.W_i.bias",
            f"{hf_prefix}.mlstm_layer.fgate_preact.weight": f"{mad_prefix}.W_f.weight",
            f"{hf_prefix}.mlstm_layer.fgate_preact.bias": f"{mad_prefix}.W_f.bias",
            f"{hf_prefix}.mlstm_layer.multihead_norm.weight": f"{mad_prefix}.mhln.weight",
        }

        for hf_key, mad_key in mlstm_mapping.items():
            if hf_key in hf_weights:
                mad_weights[mad_key] = hf_weights[hf_key]

        # Norms
        if f"{hf_prefix}.norm_mlstm.weight" in hf_weights:
            mad_weights[f"{mad_prefix}.norm.weight"] = hf_weights[f"{hf_prefix}.norm_mlstm.weight"]
        if f"{hf_prefix}.norm_ffn.weight" in hf_weights:
            mad_weights[f"{mad_prefix}.norm2.weight"] = hf_weights[f"{hf_prefix}.norm_ffn.weight"]

        # FFN - weight_loader expects up_l_proj (gate) and up_r_proj (up)
        if f"{hf_prefix}.ffn.proj_up_gate.weight" in hf_weights:
            mad_weights[f"{mad_prefix}.up_l_proj.weight"] = hf_weights[f"{hf_prefix}.ffn.proj_up_gate.weight"]
        if f"{hf_prefix}.ffn.proj_up.weight" in hf_weights:
            mad_weights[f"{mad_prefix}.up_r_proj.weight"] = hf_weights[f"{hf_prefix}.ffn.proj_up.weight"]
        if f"{hf_prefix}.ffn.proj_down.weight" in hf_weights:
            mad_weights[f"{mad_prefix}.down_proj.weight"] = hf_weights[f"{hf_prefix}.ffn.proj_down.weight"]

        if (i + 1) % 8 == 0:
            print(f"✓ Converted blocks 0-{i}")

    # Output norm
    if "backbone.out_norm.weight" in hf_weights:
        mad_weights["out_norm.weight"] = hf_weights["backbone.out_norm.weight"]
        print("✓ Mapped output norm (backbone.out_norm)")
    else:
        print("WARNING: backbone.out_norm.weight not found!")

    # LM head
    if "lm_head.weight" in hf_weights:
        mad_weights["head.W"] = hf_weights["lm_head.weight"]
        print("✓ Mapped lm_head")

    return mad_weights


def main():
    import sys

    if len(sys.argv) != 3:
        print("Usage: python convert_xlstm_7b_weights.py <hf_model_dir> <output_npz>")
        print()
        print("Example:")
        print("  python scripts/convert_xlstm_7b_weights.py \\")
        print("    model_cache/models--NX-AI--xLSTM-7b/snapshots/9dc507bd0939cf372a4a4f667335651d8e49dddb \\")
        print("    model_cache/xlstm_7b_mlx_converted.npz")
        sys.exit(1)

    hf_model_dir = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Converting xLSTM-7B weights...")
    print(f"  Input: {hf_model_dir}")
    print(f"  Output: {output_path}")
    print()

    # Load HF weights
    hf_weights = load_hf_safetensors(hf_model_dir)

    # Convert to MAD format
    print("\nConverting to MAD format...")
    mad_weights = convert_to_mad_format(hf_weights)

    # Save
    print(f"\nSaving {len(mad_weights)} tensors to {output_path}...")
    mx.savez(output_path, **mad_weights)

    print("\n✅ Conversion complete!")
    print(f"Total tensors: {len(mad_weights)}")


if __name__ == "__main__":
    main()
