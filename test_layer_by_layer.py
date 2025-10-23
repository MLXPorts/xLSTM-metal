#!/usr/bin/env python
"""
Layer-by-layer comparison between MLX and PyTorch to pinpoint divergence.
"""

import torch
import mlx.core as mx
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_arrays(mlx_arr, torch_arr, name):
    """Quick comparison helper."""
    mlx_np = np.array(mlx_arr)
    torch_np = torch_arr.detach().cpu().numpy()

    abs_diff = np.abs(mlx_np - torch_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    passed = max_diff < 1e-3

    status = "✅" if passed else "❌"
    print(f"{status} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    if not passed and abs_diff.size < 100:
        print(f"   MLX: {mlx_np.flatten()[:5]}")
        print(f"   PyT: {torch_np.flatten()[:5]}")

    return passed


print("="*80)
print("Layer-by-Layer Comparison: MLX vs PyTorch")
print("="*80)

# Load models
model_path = "xlstm_7b_model"
print("\nLoading PyTorch model...")
torch_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
torch_model.eval()

print("Loading MLX model...")
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
mlx_runner = xLSTM7BRunner()
mlx_runner.load_weights(model_path)

# Test input
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = "The capital"
inputs = tokenizer(text, return_tensors="pt")
input_ids_torch = inputs["input_ids"]
input_ids_mlx = mx.array(input_ids_torch.numpy())

print(f"\nTest input: '{text}'")
print(f"Input IDs: {input_ids_torch.squeeze().tolist()}")

# ============================================================================
# Step 1: Embedding
# ============================================================================
print("\n" + "="*80)
print("Step 1: Embedding Layer")
print("="*80)

with torch.no_grad():
    torch_emb = torch_model.backbone.embeddings(input_ids_torch)

mlx_emb = mlx_runner.model.blocks['embedding'](input_ids_mlx)

print(f"PyTorch embedding: shape={torch_emb.shape}, mean={torch_emb.mean():.4f}, std={torch_emb.std():.4f}")
print(f"MLX embedding: shape={mlx_emb.shape}, mean={float(mx.mean(mlx_emb)):.4f}, std={float(mx.std(mlx_emb)):.4f}")

compare_arrays(mlx_emb, torch_emb, "Embedding output")

# ============================================================================
# Step 2: First xLSTM block
# ============================================================================
print("\n" + "="*80)
print("Step 2: First xLSTM Block (block 0)")
print("="*80)

with torch.no_grad():
    torch_block0_out = torch_model.backbone.blocks[0](torch_emb)[0]  # (output, state)

# MLX block 0
x_mlx = mlx_emb
x_mlx, _ = mlx_runner.model.blocks['xlstm_0'](x_mlx, None)

print(f"PyTorch block 0: shape={torch_block0_out.shape}, mean={torch_block0_out.mean():.4f}, std={torch_block0_out.std():.4f}")
print(f"MLX block 0: shape={x_mlx.shape}, mean={float(mx.mean(x_mlx)):.4f}, std={float(mx.std(x_mlx)):.4f}")

compare_arrays(x_mlx, torch_block0_out, "Block 0 output")

# ============================================================================
# Step 3: Second xLSTM block
# ============================================================================
print("\n" + "="*80)
print("Step 3: Second xLSTM Block (block 1)")
print("="*80)

with torch.no_grad():
    torch_block1_out = torch_model.backbone.blocks[1](torch_block0_out)[0]

x_mlx, _ = mlx_runner.model.blocks['xlstm_1'](x_mlx, None)

print(f"PyTorch block 1: shape={torch_block1_out.shape}, mean={torch_block1_out.mean():.4f}, std={torch_block1_out.std():.4f}")
print(f"MLX block 1: shape={x_mlx.shape}, mean={float(mx.mean(x_mlx)):.4f}, std={float(mx.std(x_mlx)):.4f}")

compare_arrays(x_mlx, torch_block1_out, "Block 1 output")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Summary")
print("="*80)
print("If blocks match: The issue is in later blocks or LM head")
print("If blocks diverge: The issue is in the mLSTM/FFN implementation")
