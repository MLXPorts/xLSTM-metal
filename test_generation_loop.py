#!/usr/bin/env python
"""
Test the exact generation loop pattern
"""

import mlx.core as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("Generation Loop Pattern Test")
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

    # Manual generation loop (matching generate() method exactly)
    print("\n4. Testing manual generation loop...")
    try:
        prompt = "Hello"
        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
        max_tokens = 5

        print(f"   Prompt: '{prompt}'")
        print(f"   Token IDs: {prompt_ids}")

        # Reset state for new generation
        runner.reset_state()

        # Convert prompt to array [1, S]
        generated = list(prompt_ids)
        current_ids = mx.array([prompt_ids])

        # Generate tokens
        for i in range(max_tokens):
            print(f"\n   Iteration {i+1}:")
            print(f"     current_ids shape: {current_ids.shape}")
            print(f"     current_ids: {current_ids[0].tolist()}")

            next_token = runner.generate_next_token(
                current_ids,
                temperature=1.0,
                top_k=50,
                top_p=None
            )

            print(f"     Generated token: {next_token}")
            generated.append(next_token)

            # Update input for next iteration (only use last token for stateful generation)
            current_ids = mx.array([[next_token]])

        # Decode
        output_text = tokenizer.decode(generated)
        print(f"\n   ✓ Generated {len(generated) - len(prompt_ids)} new tokens")
        print(f"   Full output: {output_text}")

    except Exception as e:
        print(f"\n   ✗ Error at iteration {i+1}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("Generation loop test passed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
