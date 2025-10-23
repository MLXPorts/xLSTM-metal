#!/usr/bin/env python
"""
Track memory usage during generation to find resource leaks
"""

import mlx.core as mx
from mlx.core import metal
from pathlib import Path
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("Memory Tracking Test")
    print("="*80)

    model_dir = Path("xlstm_7b_model")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print("   ✓ Tokenizer loaded")

    # Create runner
    print("\n2. Creating xLSTM-7B runner...")
    runner = xLSTM7BRunner(
        embedding_dim=4096,
        num_heads=8,
        num_blocks=32,
        vocab_size=50304,
        output_logit_soft_cap=30.0
    )
    print("   ✓ Runner created")

    # Load weights
    print("\n3. Loading weights...")
    runner.load_weights(str(model_dir))
    print("   ✓ Weights loaded")

    # Print initial memory
    print("\n4. Initial memory state:")
    print(f"   Active: {metal.get_active_memory() / 1024**3:.3f} GB")
    print(f"   Peak:   {metal.get_peak_memory() / 1024**3:.3f} GB")
    print(f"   Cache:  {metal.get_cache_memory() / 1024**3:.3f} GB")

    # Run generation with memory tracking
    print("\n5. Running generation with memory tracking...")
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt)
    print(f"   Prompt: '{prompt}' ({len(input_ids)} tokens)")

    # Reset state
    runner.reset_state()

    # Convert prompt to array
    current_ids = mx.array([input_ids])

    # Generate tokens one by one with memory tracking
    max_tokens = 30
    for i in range(max_tokens):
        print(f"\n   Token {i+1}/{max_tokens}:")

        # Generate next token
        try:
            next_token = runner.generate_next_token(
                current_ids,
                temperature=0.0
            )
            print(f"     Generated: {next_token}")

            # Check memory after generation
            active_mb = metal.get_active_memory() / 1024**2
            peak_mb = metal.get_peak_memory() / 1024**2
            cache_mb = metal.get_cache_memory() / 1024**2

            print(f"     Active: {active_mb:.1f} MB, Peak: {peak_mb:.1f} MB, Cache: {cache_mb:.1f} MB")

            # Update input
            current_ids = mx.array([[next_token]])

        except Exception as e:
            print(f"\n   ✗ GPU fault at token {i+1}: {e}")
            print(f"\n   Final memory state:")
            print(f"     Active: {metal.get_active_memory() / 1024**2:.1f} MB")
            print(f"     Peak:   {metal.get_peak_memory() / 1024**2:.1f} MB")
            print(f"     Cache:  {metal.get_cache_memory() / 1024**2:.1f} MB")
            break

    print("\n" + "="*80)
    print("Memory tracking complete!")
    print("="*80)


if __name__ == "__main__":
    main()
