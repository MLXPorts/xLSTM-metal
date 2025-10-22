#!/usr/bin/env python
"""
Debug xLSTM-7B generation - check logits
"""

import mlx.core as mx
from mad.inference.text_generator import xLSTM7BTextGenerator

print("Initializing...")
generator = xLSTM7BTextGenerator(debug=False)
generator.load_weights("model_cache/xlstm_7b_mlx_converted.npz")

# Test with simple prompt
prompt = "The capital of France is"
print(f"\nPrompt: '{prompt}'")

# Encode
prompt_ids = generator.tokenizer.encode(prompt)
print(f"Token IDs: {prompt_ids.tolist()}")

# Prepend BOS token (CRITICAL - from HF README)
bos_id = generator.tokenizer.bos_token_id
print(f"BOS token ID: {bos_id}")
bos_tensor = mx.array([bos_id])
prompt_ids_with_bos = mx.concatenate([bos_tensor, prompt_ids])
print(f"Token IDs with BOS: {prompt_ids_with_bos.tolist()}")

# Decode back
decoded = generator.tokenizer.decode(prompt_ids_with_bos)
print(f"Decoded: '{decoded}'")

# Forward pass
prompt_ids_batched = mx.expand_dims(prompt_ids_with_bos, 0)
logits, state = generator.model(prompt_ids_batched, None)

print(f"\nLogits shape: {logits.shape}")

# Apply soft cap (CRITICAL - canonical code does this!)
from mad.blocks.mlstm_mlx.components import soft_cap
logits_capped = soft_cap(logits, generator.output_logit_soft_cap)
print(f"Applied soft cap with value {generator.output_logit_soft_cap}")

print(f"Logits for last token:")

# Get top 10 tokens
last_logits = logits_capped[0, -1, :]
top_k = 10
top_indices = mx.argsort(last_logits)[-top_k:][::-1]
top_logits = last_logits[top_indices]

print("\nTop 10 predictions:")
for i, (idx, logit) in enumerate(zip(top_indices.tolist(), top_logits.tolist())):
    token_text = generator.tokenizer.decode([idx])
    print(f"  {i+1}. Token {idx:5d} (logit={logit:7.2f}): '{token_text}'")
