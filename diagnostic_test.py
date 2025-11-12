#!/usr/bin/env python3
"""Quick diagnostic to check if model loads and first forward pass works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from xlstm_metal.mlx_jit.models.wired_xlstm import WiredxLSTM

print("Loading model...")
model = WiredxLSTM.from_pretrained(
    "xlstm_7b_model",
    compute_dtype=mx.float32,  # Changed from bfloat16 to match config
    state_dtype=mx.float32,
    norm_reduce_force_float32=True
)

print("✓ Model loaded")
print(f"Blocks: {len(model.blocks)}")
print(f"First block type: {type(model.blocks[0])}")

# Check if weights look reasonable
print("\nChecking first block norm weights...")
first_norm_weight = model.blocks[0].norm_mlstm.weight
print(f"  norm_mlstm weight shape: {first_norm_weight.shape}")
print(f"  norm_mlstm weight dtype: {first_norm_weight.dtype}")
print(f"  norm_mlstm weight min/max: {mx.min(first_norm_weight).item():.6f} / {mx.max(first_norm_weight).item():.6f}")
print(f"  norm_mlstm has NaN: {mx.any(mx.isnan(first_norm_weight)).item()}")
print(f"  norm_mlstm force_float32: {model.blocks[0].norm_mlstm.force_float32}")

print("\nChecking first block FFN weights...")
ffn_up = model.blocks[0].ffn_proj_up.weight
ffn_gate = model.blocks[0].ffn_proj_up_gate.weight
ffn_down = model.blocks[0].ffn_proj_down.weight
print(f"  ffn_proj_up weight shape: {ffn_up.shape}, dtype: {ffn_up.dtype}")
print(f"  ffn_proj_up min/max: {mx.min(ffn_up).item():.6f} / {mx.max(ffn_up).item():.6f}")
print(f"  ffn_proj_up has NaN: {mx.any(mx.isnan(ffn_up)).item()}")
print(f"  ffn_proj_up_gate weight shape: {ffn_gate.shape}, dtype: {ffn_gate.dtype}")
print(f"  ffn_proj_up_gate min/max: {mx.min(ffn_gate).item():.6f} / {mx.max(ffn_gate).item():.6f}")
print(f"  ffn_proj_down weight shape: {ffn_down.shape}, dtype: {ffn_down.dtype}")
print(f"  ffn_proj_down min/max: {mx.min(ffn_down).item():.6f} / {mx.max(ffn_down).item():.6f}")

print("\nChecking embedding weights...")
if model.embedding is not None:
    emb_weight = model.embedding.weight
    print(f"  embedding weight shape: {emb_weight.shape}, dtype: {emb_weight.dtype}")
    print(f"  embedding min/max: {mx.min(emb_weight).item():.6f} / {mx.max(emb_weight).item():.6f}")
    print(f"  embedding has NaN: {mx.any(mx.isnan(emb_weight)).item()}")
    print(f"  embedding has Inf: {mx.any(mx.isinf(emb_weight)).item()}")

# Try a simple forward pass
print("\nTrying forward pass with dummy input...")
dummy_input = mx.array([[1, 2, 3, 4]], dtype=mx.int32)  # Shape: [1, 4]
try:
    output = model(dummy_input)
    print(f"✓ Forward pass succeeded! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Try with tokenizer
print("\nTrying with actual tokenizer...")
from xlstm_metal.mlx_jit.tokenizer import TokenizerBlock, TokenizerConfig

tokenizer_config = TokenizerConfig(model_path="xlstm_7b_model")
tokenizer = TokenizerBlock(tokenizer_config)

test_prompt = "Hello world, how are you?"
print(f"Test prompt: '{test_prompt}'")
prompt_ids = tokenizer.encode(test_prompt)
print(f"Encoded shape: {prompt_ids.shape}, dtype: {prompt_ids.dtype}")
print(f"Token IDs: {prompt_ids.tolist()}")
print(f"Number of tokens: {len(prompt_ids.tolist())}")

# Convert to proper shape [B, S]
if prompt_ids.ndim == 1:
    prompt_ids = mx.expand_dims(prompt_ids, axis=0)

print(f"Reshaped to: {prompt_ids.shape}")

try:
    output = model(prompt_ids)
    print(f"✓ Forward pass with tokenizer succeeded! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass with tokenizer failed: {e}")
    import traceback
    traceback.print_exc()

