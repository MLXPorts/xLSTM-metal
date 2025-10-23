#!/usr/bin/env python
"""
Minimal debug test for GPU fault isolation
"""

import mlx.core as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("Minimal GPU Fault Debug Test")
    print("=" * 80)

    model_dir = Path("xlstm_7b_model")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"   ✓ Tokenizer loaded")

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
    print(f"   ✓ Weights loaded")

    # Test 1: Single token forward pass
    print("\n4. Testing single token forward pass...")
    try:
        # Just BOS token
        input_ids = mx.array([[tokenizer.bos_token_id]])
        print(f"   Input shape: {input_ids.shape}")

        # Forward pass
        print("   Running forward pass...")
        logits, state = runner.forward(input_ids, None)
        print(f"   Logits shape (before eval): {logits.shape}")

        # Evaluate
        print("   Evaluating logits...")
        mx.eval(logits)
        print(f"   ✓ Logits evaluated: {logits.shape}")

        # Check state
        if state is not None:
            print(f"   Evaluating state ({len(state)} blocks)...")
            for block_name, state_tuple in state.items():
                if state_tuple is not None and isinstance(state_tuple, tuple):
                    c, n, m = state_tuple
                    print(f"     Block {block_name}: c={c.shape}, n={n.shape}, m={m.shape}")
                    mx.eval(c, n, m)
            print("   ✓ State evaluated")

        print("   ✓ Single token forward pass successful!")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 2: Generate 1 token
    print("\n5. Testing 1 token generation...")
    try:
        runner.reset_state()
        input_ids = mx.array([[tokenizer.bos_token_id]])

        print("   Generating next token...")
        next_token = runner.generate_next_token(input_ids, temperature=1.0, top_k=50)
        print(f"   ✓ Next token: {next_token}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 3: Generate 2 tokens (autoregressive)
    print("\n6. Testing 2 token generation (autoregressive)...")
    try:
        runner.reset_state()

        # First token (BOS)
        input_ids = mx.array([[tokenizer.bos_token_id]])
        print(f"   Step 1: Input {input_ids[0].tolist()}")
        next_token_1 = runner.generate_next_token(input_ids, temperature=1.0, top_k=50)
        print(f"   Generated: {next_token_1}")

        # Second token (use generated token)
        input_ids = mx.array([[next_token_1]])
        print(f"   Step 2: Input {input_ids[0].tolist()}")
        next_token_2 = runner.generate_next_token(input_ids, temperature=1.0, top_k=50)
        print(f"   Generated: {next_token_2}")

        print("   ✓ 2 token generation successful!")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("All minimal tests passed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
