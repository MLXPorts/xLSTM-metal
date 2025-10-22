#!/usr/bin/env python
"""
Test xLSTM-7B Text Generation with Tokenizer Block
"""

from mad.inference.text_generator import xLSTM7BTextGenerator

print("=" * 80)
print("xLSTM-7B Text Generation Test")
print("=" * 80)

# Initialize generator with debug mode
print("\n1. Initializing text generator...")
generator = xLSTM7BTextGenerator(debug=True)

# Load weights
print("\n2. Loading pretrained weights...")
generator.load_weights("model_cache/xlstm_7b_mlx_converted.npz")

# Test generation
print("\n3. Testing text generation...")
print("=" * 80)

prompt = "The capital of France is"
print(f"Prompt: {prompt}")
print("-" * 80)

output = generator.generate(
    prompt,
    max_tokens=20,
    temperature=0.8,
    stream=True
)

print("=" * 80)
print(f"\nFull output:\n{output}")
print("\n" + "=" * 80)
print("✅ Text generation successful!")
print("=" * 80)
