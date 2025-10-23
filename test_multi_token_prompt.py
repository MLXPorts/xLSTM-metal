#!/usr/bin/env python
"""
Test multi-token prompt to isolate GPU fault
"""

import mlx.core as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("Multi-Token Prompt Test")
    print("=" * 80)

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

    # Test with multi-token prompt
    print("\n4. Testing multi-token prompt forward pass...")
    try:
        prompt = "Hello"
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
        print(f"   Prompt: '{prompt}'")
        print(f"   Token IDs: {input_ids}")
        print(f"   Length: {len(input_ids)} tokens")

        # Forward pass with multiple tokens
        input_array = mx.array([input_ids])  # [1, S] where S = len(input_ids)
        print(f"   Input shape: {input_array.shape}")

        print("   Running forward pass...")
        logits, state = runner.forward(input_array, None)

        print("   Evaluating logits...")
        mx.eval(logits)
        print(f"   ✓ Logits evaluated: {logits.shape}")

        if state is not None:
            print(f"   Evaluating state...")
            for block_name, state_tuple in state.items():
                if state_tuple is not None and isinstance(state_tuple, tuple):
                    c, n, m = state_tuple
                    mx.eval(c, n, m)
            print("   ✓ State evaluated")

        print("   ✓ Multi-token prompt forward pass successful!")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test generate_next_token with multi-token prompt
    print("\n5. Testing generate_next_token with multi-token prompt...")
    try:
        runner.reset_state()

        prompt = "Hello"
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
        input_array = mx.array([input_ids])

        print(f"   Generating from prompt: {input_ids}")
        next_token = runner.generate_next_token(input_array, temperature=1.0, top_k=50)
        print(f"   ✓ Next token: {next_token}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("All multi-token prompt tests passed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
