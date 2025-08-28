#!/usr/bin/env python
"""Check the structure of the pretrained weights"""

import json
from safetensors import safe_open

model_path = "/Volumes/emberstuff/xLSTM/xlstm_7b_model"

# Load model index
with open(f"{model_path}/model.safetensors.index.json", 'r') as f:
    index = json.load(f)

# Get unique layer names
weight_keys = list(index['weight_map'].keys())

print("Sample weight keys from pretrained model:")
print("=" * 50)

# Group by prefix
prefixes = {}
for key in weight_keys[:100]:  # First 100 keys
    parts = key.split('.')
    prefix = '.'.join(parts[:2]) if len(parts) > 1 else parts[0]
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(key)

for prefix, keys in sorted(prefixes.items())[:20]:
    print(f"\n{prefix}:")
    for key in keys[:3]:
        print(f"  {key}")

print("\n" + "=" * 50)
print("\nOur model structure needs:")
print("  embedding.weight")
print("  backbone.blocks.{i}.norm_mlstm.weight")
print("  backbone.blocks.{i}.mlstm_layer.q.weight")
print("  backbone.blocks.{i}.mlstm_layer.k.weight")
print("  backbone.blocks.{i}.mlstm_layer.v.weight")
print("  backbone.blocks.{i}.ffn.proj_up.weight")
print("  lm_head.weight")
print("\nPretrained model has:")
print("  model.embedding.weight")
print("  model.xlstm_block_stack.blocks.{i}...")

# Check actual tensor shapes
print("\n" + "=" * 50)
print("Checking tensor shapes:")
with safe_open(f"{model_path}/model-00001-of-00006.safetensors", framework="pt", device="cpu") as f:
    for key in list(f.keys())[:10]:
        tensor = f.get_tensor(key)
        print(f"  {key}: {tensor.shape}")