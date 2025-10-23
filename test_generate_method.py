#!/usr/bin/env python
"""
Test the generate() method directly
"""

import mlx.core as mx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("Generate Method Test")
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

    # Test generate() method with different lengths
    test_lengths = [3, 5, 10]

    for max_tokens in test_lengths:
        print(f"\n4. Testing generate() with max_tokens={max_tokens}...")
        try:
            prompt = "Hello"
            prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)

            print(f"   Prompt: '{prompt}' ({len(prompt_ids)} tokens)")

            # Call generate() method
            output_ids = runner.generate(
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=1.0,
                top_k=50,
                top_p=None,
                stop_tokens=[tokenizer.eos_token_id]
            )

            # Decode
            output_text = tokenizer.decode(output_ids)
            print(f"   ✓ Generated {len(output_ids) - len(prompt_ids)} new tokens")
            print(f"   Output: {output_text[:100]}...")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 80)
    print("All generate() tests passed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
