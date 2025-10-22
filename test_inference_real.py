#!/usr/bin/env python
"""
Test xLSTM-7B Inference with Real Pretrained Weights
"""

import mlx.core as mx
from mad.wiring.mlx import create_xlstm_7b_wiring, WiredMADModel
from mad.utils.weight_loader import load_weights_into_wired_model
from mad.blocks.mlstm_mlx.components import soft_cap

print("=" * 80)
print("xLSTM-7B Real Inference Test")
print("=" * 80)

# Create wiring
print("\n1. Creating xLSTM-7B wiring...")
wiring = create_xlstm_7b_wiring()
print(f"   ✓ Wiring created with {wiring.num_blocks} blocks")

# Create model
print("\n2. Creating WiredMADModel...")
model = WiredMADModel(wiring, 'embedding', 'lm_head')
print(f"   ✓ Model created")

# Load pretrained weights
print("\n3. Loading pretrained weights...")
npz_path = "model_cache/xlstm_7b_mlx_converted.npz"
load_weights_into_wired_model(npz_path, model)
print(f"   ✓ Weights loaded from {npz_path}")

# Test forward pass with random input
print("\n4. Testing forward pass...")
batch_size = 1
seq_len = 10
input_ids = mx.random.randint(0, 50304, (batch_size, seq_len))

print(f"   Input shape: {input_ids.shape}")
logits, state = model(input_ids)

# Apply soft cap
logits_capped = soft_cap(logits, 30.0)

print(f"   Output logits shape: {logits_capped.shape}")
print(f"   Logits range: [{float(logits_capped.min()):.2f}, {float(logits_capped.max()):.2f}]")
print(f"   State blocks: {len(state)}")

# Generate next token
print("\n5. Testing token generation...")
next_token_logits = logits_capped[0, -1, :]
probs = mx.softmax(next_token_logits, axis=-1)
next_token = mx.argmax(probs)

print(f"   Next token: {int(next_token)}")
print(f"   Top-5 tokens: {mx.argsort(probs)[-5:][::-1].tolist()}")

print("\n" + "=" * 80)
print("✅ Real inference test successful!")
print("=" * 80)
