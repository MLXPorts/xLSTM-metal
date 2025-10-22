#!/usr/bin/env python
"""
Test stateless generation (no state passing)
"""

import mlx.core as mx
from mad.inference.text_generator import xLSTM7BTextGenerator

print("Initializing...")
generator = xLSTM7BTextGenerator(debug=False)
generator.load_weights("model_cache/xlstm_7b_mlx_converted.npz")

# Test with simple prompt
prompt = "The capital of France is Paris"
print(f"\nPrompt: '{prompt}'")

# Encode
prompt_ids = generator.tokenizer.encode(prompt)
print(f"Token IDs: {prompt_ids.tolist()}")
prompt_ids_batched = mx.expand_dims(prompt_ids, 0)

# Forward pass WITHOUT state
logits, _ = generator.model(prompt_ids_batched, None)
mx.eval(logits)  # Force evaluation

print(f"\nLogits shape: {logits.shape}")

# Check each token's top prediction
for i in range(prompt_ids.shape[0]):
    token_logits = logits[0, i, :]
    top_idx = int(mx.argmax(token_logits))
    top_token = generator.tokenizer.decode([top_idx])
    actual_token = generator.tokenizer.decode([int(prompt_ids[i])])
    print(f"Position {i}: actual='{actual_token}' top_pred='{top_token}'")
