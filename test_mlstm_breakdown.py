#!/usr/bin/env python
"""
Break down mLSTM layer step by step to find divergence point.
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
    print(f"{status} {name}")
    print(f"   max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    if not passed or name.startswith("FINAL"):
        print(f"   MLX: shape={mlx_arr.shape}, mean={float(mx.mean(mlx_arr)):.6f}, std={float(mx.std(mlx_arr)):.6f}")
        print(f"   PyT: shape={torch_arr.shape}, mean={torch_arr.mean():.6f}, std={torch_arr.std():.6f}")

    return passed


print("="*80)
print("mLSTM Layer Breakdown")
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

# Get embeddings and norm
with torch.no_grad():
    torch_emb = torch_model.backbone.embeddings(input_ids_torch)
    torch_x = torch_model.backbone.blocks[0].norm_mlstm(torch_emb)

mlx_emb = mlx_runner.model.blocks['embedding'](input_ids_mlx)
mlx_x = mlx_runner.model.blocks['xlstm_0'].xlstm_norm(mlx_emb)

print(f"\nInput to mLSTM: '{text}'")
print(f"After RMSNorm: shape={mlx_x.shape}")

# Get mLSTM layers
torch_mlstm = torch_model.backbone.blocks[0].mlstm_layer
mlx_mlstm = mlx_runner.model.blocks['xlstm_0'].xlstm

print("\n" + "="*80)
print("mLSTM Internal Steps")
print("="*80)

# Step 1: QKV projections
print("\n--- Step 1: QKV Projections ---")
with torch.no_grad():
    torch_q = torch_mlstm.q(torch_x)
    torch_k = torch_mlstm.k(torch_x)
    torch_v = torch_mlstm.v(torch_x)

mlx_q = mlx_mlstm.q(mlx_x)
mlx_k = mlx_mlstm.k(mlx_x)
mlx_v = mlx_mlstm.v(mlx_x)

compare_arrays(mlx_q, torch_q, "Q projection", atol=1e-4)
compare_arrays(mlx_k, torch_k, "K projection", atol=1e-4)
compare_arrays(mlx_v, torch_v, "V projection", atol=1e-4)

# Step 2: Gate projections
print("\n--- Step 2: Gate Projections ---")
with torch.no_grad():
    torch_i_pre = torch_mlstm.igate_preact(torch_x)
    torch_f_pre = torch_mlstm.fgate_preact(torch_x)
    torch_o_pre = torch_mlstm.ogate_preact(torch_x)

mlx_i_pre = mlx_mlstm.igate_preact(mlx_x)
mlx_f_pre = mlx_mlstm.fgate_preact(mlx_x)
mlx_o_pre = mlx_mlstm.ogate_preact(mlx_x)

compare_arrays(mlx_i_pre, torch_i_pre, "i_gate preact", atol=1e-4)
compare_arrays(mlx_f_pre, torch_f_pre, "f_gate preact", atol=1e-4)
compare_arrays(mlx_o_pre, torch_o_pre, "o_gate preact", atol=1e-4)

# Step 3: Soft-cap gates
print("\n--- Step 3: Soft-cap Gates (15.0) ---")
with torch.no_grad():
    cap = 15.0
    torch_i_capped = cap * torch.tanh(torch_i_pre / cap)
    torch_f_capped = cap * torch.tanh(torch_f_pre / cap)

from xlstm_metal.blocks.mlstm_mlx.components import soft_cap
mlx_i_capped = soft_cap(mlx_i_pre, 15.0)
mlx_f_capped = soft_cap(mlx_f_pre, 15.0)

compare_arrays(mlx_i_capped, torch_i_capped, "i_gate soft-capped", atol=1e-5)
compare_arrays(mlx_f_capped, torch_f_capped, "f_gate soft-capped", atol=1e-5)

print("\n--- Step 4: Reshape for Multi-head ---")
B, S, _ = mlx_q.shape
NH = 8
QK_DH = 256  # 2048 / 8
V_DH = 512   # 4096 / 8

# PyTorch reshaping
with torch.no_grad():
    torch_q_mh = torch_q.reshape(B, S, NH, QK_DH).transpose(1, 2)  # [B, NH, S, QK_DH]
    torch_k_mh = torch_k.reshape(B, S, NH, QK_DH).transpose(1, 2)
    torch_v_mh = torch_v.reshape(B, S, NH, V_DH).transpose(1, 2)
    torch_i_mh = torch_i_capped.transpose(1, 2)  # [B, NH, S]
    torch_f_mh = torch_f_capped.transpose(1, 2)

# MLX reshaping
mlx_q_mh = mlx_q.reshape(B, S, NH, QK_DH).transpose(0, 2, 1, 3)
mlx_k_mh = mlx_k.reshape(B, S, NH, QK_DH).transpose(0, 2, 1, 3)
mlx_v_mh = mlx_v.reshape(B, S, NH, V_DH).transpose(0, 2, 1, 3)
mlx_i_mh = mlx_i_capped.transpose(0, 2, 1)
mlx_f_mh = mlx_f_capped.transpose(0, 2, 1)

compare_arrays(mlx_q_mh, torch_q_mh, "Q reshaped [B,NH,S,QK_DH]", atol=1e-5)
compare_arrays(mlx_k_mh, torch_k_mh, "K reshaped [B,NH,S,QK_DH]", atol=1e-5)
compare_arrays(mlx_v_mh, torch_v_mh, "V reshaped [B,NH,S,V_DH]", atol=1e-5)
compare_arrays(mlx_i_mh, torch_i_mh, "i_gate reshaped [B,NH,S]", atol=1e-5)
compare_arrays(mlx_f_mh, torch_f_mh, "f_gate reshaped [B,NH,S]", atol=1e-5)

print("\n" + "="*80)
print("Summary")
print("="*80)
print("If all projections/reshapes match, the bug is in the mLSTM kernel math.")
print("If projections/reshapes diverge, the bug is in weight loading or layer config.")
