#!/usr/bin/env python
"""
Test greedy decoding (temperature=0.0) to check if sampling is the issue
"""

from pathlib import Path
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("xLSTM-7B Greedy Decoding Test")
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

    # Simple prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "2 + 2 =",
    ]

    print("\n4. Running greedy inference (temperature=0.0)...")
    print("="*80)

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: \"{prompt}\"")
        print("-"*80)

        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        print(f"Input IDs: {input_ids}")

        # Generate with greedy decoding
        try:
            output_ids = runner.generate(
                prompt_ids=input_ids,
                max_tokens=20,
                temperature=0.0,  # Greedy decoding
            )

            # Decode
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"Output IDs: {output_ids}")
            print(f"Generated: {output_text}\n")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("="*80)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
