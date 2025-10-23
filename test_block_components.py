#!/usr/bin/env python
"""
Test each component of xLSTM block 0 individually to pinpoint divergence.
"""

import torch
import mlx.core as mx
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_arrays(mlx_arr, torch_arr, name, rtol=1e-3, atol=1e-3):
    """Compare MLX and PyTorch arrays."""
    mlx_np = np.array(mlx_arr)
    torch_np = torch_arr.detach().cpu().numpy()

    abs_diff = np.abs(mlx_np - torch_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    passed = max_diff < atol

    status = "✅" if passed else "❌"
    print(f"{status} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    if not passed:
        print(f"   MLX: mean={float(mx.mean(mlx_arr)):.4f}, std={float(mx.std(mlx_arr)):.4f}")
        print(f"   PyT: mean={torch_arr.mean():.4f}, std={torch_arr.std():.4f}")

    return passed


print("="*80)
print("Block 0 Component-by-Component Comparison")
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

# Get embeddings
with torch.no_grad():
    torch_emb = torch_model.backbone.embeddings(input_ids_torch)
mlx_emb = mlx_runner.model.blocks['embedding'](input_ids_mlx)

print("\n" + "="*80)
print("Embeddings")
print("="*80)
compare_arrays(mlx_emb, torch_emb, "Embedding output")

# ============================================================================
# Break down Block 0
# ============================================================================
print("\n" + "="*80)
print("Block 0 Breakdown")
print("="*80)

torch_block = torch_model.backbone.blocks[0]
mlx_block = mlx_runner.model.blocks['xlstm_0']

# Step 1: xlstm_norm (RMSNorm before mLSTM)
print("\n--- Step 1: xlstm_norm (pre-mLSTM RMSNorm) ---")
with torch.no_grad():
    torch_norm1 = torch_block.norm_mlstm(torch_emb)
mlx_norm1 = mlx_block.xlstm_norm(mlx_emb)
compare_arrays(mlx_norm1, torch_norm1, "xlstm_norm output")

# Step 2: mLSTM layer (without residual)
print("\n--- Step 2: mLSTM layer ---")
with torch.no_grad():
    torch_mlstm_out, _ = torch_block.mlstm_layer(torch_norm1)
mlx_mlstm_out, _ = mlx_block.xlstm(mlx_norm1, None)
compare_arrays(mlx_mlstm_out, torch_mlstm_out, "mLSTM output (no residual)")

# Step 3: mLSTM with residual
print("\n--- Step 3: mLSTM + residual ---")
torch_mlstm_res = torch_emb + torch_mlstm_out
mlx_mlstm_res = mx.add(mlx_emb, mlx_mlstm_out)
compare_arrays(mlx_mlstm_res, torch_mlstm_res, "mLSTM + residual")

# Step 4: ffn_norm (RMSNorm before FFN)
print("\n--- Step 4: ffn_norm (pre-FFN RMSNorm) ---")
with torch.no_grad():
    torch_norm2 = torch_block.norm_ffn(torch_mlstm_res)
mlx_norm2 = mlx_block.ffn_norm(mlx_mlstm_res)
compare_arrays(mlx_norm2, torch_norm2, "ffn_norm output")

# Step 5: FFN layer (without residual)
print("\n--- Step 5: FFN layer ---")
with torch.no_grad():
    torch_ffn_out = torch_block.ffn(torch_norm2)
mlx_ffn_out = mlx_block.ffn(mlx_norm2)
compare_arrays(mlx_ffn_out, torch_ffn_out, "FFN output (no residual)")

# Step 6: FFN with residual (final block output)
print("\n--- Step 6: FFN + residual (Block 0 final) ---")
torch_block0_final = torch_mlstm_res + torch_ffn_out
mlx_block0_final = mx.add(mlx_mlstm_res, mlx_ffn_out)
compare_arrays(mlx_block0_final, torch_block0_final, "Block 0 final output")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("The first failing component is the root cause of divergence.")
