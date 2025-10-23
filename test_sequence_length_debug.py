#!/usr/bin/env python
"""
Debug test to trace sequence lengths during generation
"""

import mlx.core as mx
from pathlib import Path
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("Sequence Length Debug Test")
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

    # Test with initial prompt (2 tokens)
    print("\n4. Testing with 2-token prompt (BOS + Hello)...")
    prompt = "Hello"
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
    print(f"   Initial prompt IDs: {input_ids} (length={len(input_ids)})")

    # Reset state
    runner.reset_state()

    # Convert to array
    current_ids = mx.array([input_ids])
    print(f"   Input shape: {current_ids.shape}")

    # First forward pass (should use Metal kernel with S=2)
    print("\n5. First forward pass (initial prompt)...")
    logits, runner.state = runner.forward(current_ids, runner.state)
    mx.eval(logits)
    print(f"   Logits shape: {logits.shape}")
    print(f"   State keys: {list(runner.state.keys()) if runner.state else None}")

    # Get next token
    next_token_logits = logits[0, -1, :]
    next_token = mx.argmax(next_token_logits).item()
    print(f"   Next token: {next_token}")

    # Second forward pass (single token - should use Python MLX with S=1)
    print("\n6. Second forward pass (single token)...")
    current_ids = mx.array([[next_token]])
    print(f"   Input shape: {current_ids.shape}")

    logits, runner.state = runner.forward(current_ids, runner.state)
    mx.eval(logits)
    print(f"   Logits shape: {logits.shape}")

    # Third forward pass
    next_token = mx.argmax(logits[0, -1, :]).item()
    print(f"   Next token: {next_token}")

    print("\n7. Third forward pass (single token)...")
    current_ids = mx.array([[next_token]])
    print(f"   Input shape: {current_ids.shape}")

    logits, runner.state = runner.forward(current_ids, runner.state)
    mx.eval(logits)
    print(f"   Logits shape: {logits.shape}")

    print("\n" + "="*80)
    print("Sequence length debug complete!")
    print("="*80)


if __name__ == "__main__":
    main()
