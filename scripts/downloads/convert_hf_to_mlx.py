#!/usr/bin/env python
"""
Convert HuggingFace xLSTM weights to MLX format.
"""

import json
import sys
import os
from pathlib import Path
from safetensors import safe_open
import mlx.core as mx

# Add xlstm-solace-mlx to path
root = os.path.dirname(os.path.abspath(__file__))
mlx_src = os.path.join(root, "xlstm-solace-mlx", "src")
if mlx_src not in sys.path:
    sys.path.insert(0, mlx_src)

from xlstm_solace_mlx.api import create_xlstm_model

def load_hf_weights(model_path):
    """Load all sharded HuggingFace weights into a single dict."""
    model_path = Path(model_path)
    
    # Load the index to find all shard files
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    # Get all unique shard files
    shard_files = set(index["weight_map"].values())
    print(f"Loading {len(shard_files)} shard files...")
    
    # Load all weights
    weights = {}
    for shard_file in shard_files:
        shard_path = model_path / shard_file
        print(f"  Loading {shard_file}...")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = mx.array(tensor.numpy())
    
    return weights

def convert_hf_to_mlx_params(hf_weights, mlx_model):
    """Convert HuggingFace parameter names to MLX format."""
    mlx_params = {}
    
    # Handle embedding layer
    if "backbone.embeddings.weight" in hf_weights:
        mlx_params["embedding.weight"] = hf_weights["backbone.embeddings.weight"]
        print("Mapped: backbone.embeddings.weight -> embedding.weight")
    
    # Handle blocks
    num_blocks = mlx_model.num_layers
    for i in range(num_blocks):
        hf_prefix = f"backbone.blocks.{i}"
        mlx_prefix = f"blocks.{i}"
        
        # Map mLSTM layer parameters
        mlstm_mappings = {
            f"{hf_prefix}.mlstm_layer.q.weight": f"{mlx_prefix}.W_q.weight",
            f"{hf_prefix}.mlstm_layer.k.weight": f"{mlx_prefix}.W_k.weight", 
            f"{hf_prefix}.mlstm_layer.v.weight": f"{mlx_prefix}.W_v.weight",
            f"{hf_prefix}.mlstm_layer.out_proj.weight": f"{mlx_prefix}.W_o.weight",
            f"{hf_prefix}.mlstm_layer.igate_preact.weight": f"{mlx_prefix}.W_i.weight",
            f"{hf_prefix}.mlstm_layer.igate_preact.bias": f"{mlx_prefix}.W_i.bias",
            f"{hf_prefix}.mlstm_layer.fgate_preact.weight": f"{mlx_prefix}.W_f.weight",
            f"{hf_prefix}.mlstm_layer.fgate_preact.bias": f"{mlx_prefix}.W_f.bias",
            f"{hf_prefix}.mlstm_layer.ogate_preact.weight": f"{mlx_prefix}.W_o.weight",
            f"{hf_prefix}.norm_mlstm.weight": f"{mlx_prefix}.norm.weight",
            f"{hf_prefix}.norm_ffn.weight": f"{mlx_prefix}.norm2.weight",
            f"{hf_prefix}.mlstm_layer.multihead_norm.weight": f"{mlx_prefix}.mhln.weight",
        }
        
        for hf_key, mlx_key in mlstm_mappings.items():
            if hf_key in hf_weights:
                mlx_params[mlx_key] = hf_weights[hf_key]
                print(f"Mapped: {hf_key} -> {mlx_key}")
        
        # Map FFN parameters  
        ffn_mappings = {
            f"{hf_prefix}.ffn.proj_up.weight": f"{mlx_prefix}.up_l_proj.weight",
            f"{hf_prefix}.ffn.proj_up_gate.weight": f"{mlx_prefix}.up_r_proj.weight", 
            f"{hf_prefix}.ffn.proj_down.weight": f"{mlx_prefix}.down_proj.weight",
        }
        
        for hf_key, mlx_key in ffn_mappings.items():
            if hf_key in hf_weights:
                mlx_params[mlx_key] = hf_weights[hf_key]
                print(f"Mapped: {hf_key} -> {mlx_key}")
    
    # Handle output head - look for both possible names
    head_keys = ["lm_head.weight", "head.weight", "output.weight"]
    for key in head_keys:
        if key in hf_weights:
            mlx_params["head.W"] = hf_weights[key]
            print(f"Mapped: {key} -> head.W")
            break
    
    return mlx_params

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_hf_to_mlx.py <hf_model_path> <output_mlx_path>")
        sys.exit(1)
    
    hf_model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Loading HuggingFace model from: {hf_model_path}")
    hf_weights = load_hf_weights(hf_model_path)
    
    print(f"Creating MLX model...")
    # Use config from the HF model
    config_path = Path(hf_model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    mlx_model = create_xlstm_model(
        vocab_size=config["vocab_size"],
        num_layers=config["num_blocks"], 
        signature=(1, 1),  # Assume mLSTM only for now
        inp_dim=config["embedding_dim"],
        head_dim=config["embedding_dim"] // config["num_heads"],
        head_num=config["num_heads"]
    )
    
    print("Converting parameters...")
    mlx_params = convert_hf_to_mlx_params(hf_weights, mlx_model)
    
    print(f"Saving converted weights to: {output_path}")
    mx.savez(output_path, **mlx_params)
    
    print("Conversion complete!")
    print(f"Converted {len(mlx_params)} parameters")

if __name__ == "__main__":
    main()
