#!/usr/bin/env python
"""
Test multiple prompts in sequence
"""

import mlx.core as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("Multiple Prompts Test")
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

    # Test prompts (same as test_real_inference.py)
    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "Machine learning is"
    ]

    print("\n4. Testing multiple prompts with max_tokens=10...")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Test {i}/{len(prompts)}: \"{prompt}\"")

        try:
            # Tokenize and prepend BOS token
            input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
            print(f"     Tokenized: {len(input_ids)} tokens (including BOS)")

            # Generate
            print("     Generating...")
            output_ids = runner.generate(
                prompt_ids=input_ids,
                max_tokens=10,
                temperature=1.0,
                top_k=50,
                stop_tokens=[tokenizer.eos_token_id]
            )

            # Decode
            output_text = tokenizer.decode(output_ids)
            print(f"     Output: {output_text[:80]}...")
            print(f"     Generated {len(output_ids) - len(input_ids)} new tokens")

        except Exception as e:
            print(f"     ✗ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n5. Testing with max_tokens=30 (like original test)...")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Test {i}/{len(prompts)}: \"{prompt}\"")

        try:
            input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)

            print("     Generating...")
            output_ids = runner.generate(
                prompt_ids=input_ids,
                max_tokens=30,
                temperature=1.0,
                top_k=50,
                stop_tokens=[tokenizer.eos_token_id]
            )

            output_text = tokenizer.decode(output_ids)
            print(f"     Output: {output_text[:80]}...")
            print(f"     Generated {len(output_ids) - len(input_ids)} new tokens")

        except Exception as e:
            print(f"     ✗ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 80)
    print("All multiple prompt tests passed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
