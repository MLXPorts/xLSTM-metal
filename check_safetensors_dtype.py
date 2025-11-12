#!/usr/bin/env python3
"""Check safetensors weight dtypes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards

print("Loading safetensors...")
weights = load_safetensor_shards("xlstm_7b_model")

print("\nChecking embedding weight dtype in safetensors:")
if 'backbone.embeddings.weight' in weights:
    emb = weights['backbone.embeddings.weight']
    print(f"  Original dtype in safetensors: {emb.dtype}")
    print(f"  Shape: {emb.shape}")
    print(f"  Min/max: {mx.min(emb).item():.6f} / {mx.max(emb).item():.6f}")

    # Try casting
    emb_fp32 = mx.array(emb, dtype=mx.float32)
    print(f"\n  After cast to float32: {emb_fp32.dtype}")
    print(f"  Min/max after cast: {mx.min(emb_fp32).item():.6f} / {mx.max(emb_fp32).item():.6f}")
else:
    print("  Embedding weight not found in safetensors!")

print("\nChecking a block weight:")
if 'backbone.blocks.0.norm_mlstm.weight' in weights:
    norm = weights['backbone.blocks.0.norm_mlstm.weight']
    print(f"  norm_mlstm dtype: {norm.dtype}")

