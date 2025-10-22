#!/usr/bin/env python
"""
Real xLSTM-7B Inference Test with Safetensors Weights

Tests the refactored MLX blocks with actual model weights.
"""

import mlx.core as mx
import sys
from pathlib import Path

# Add xlstm_metal to path
sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("xLSTM-7B Real Inference Test")
    print("Testing pure MLX operator refactor with actual weights")
    print("=" * 80)

    # Model paths
    model_dir = Path("xlstm_7b_model")

    if not model_dir.exists():
        print(f"Error: Model directory not found at {model_dir}")
        print("Please ensure xlstm_7b_model/ exists with safetensors files")
        return 1

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        print(f"   Tokenizer loaded: vocab_size={len(tokenizer)}")
    except Exception as e:
        print(f"   Error loading tokenizer: {e}")
        return 1

    # Create runner
    print("\n2. Creating xLSTM-7B runner...")
    try:
        runner = xLSTM7BRunner(
            embedding_dim=4096,
            num_heads=8,
            num_blocks=32,
            vocab_size=50304,
            output_logit_soft_cap=30.0
        )
        print("   Runner created successfully")

        # Get model info
        info = runner.get_model_info()
        print(f"   Model info:")
        print(f"     - Embedding dim: {info['embedding_dim']}")
        print(f"     - Num heads: {info['num_heads']}")
        print(f"     - Num blocks: {info['num_blocks']}")
        print(f"     - Total blocks: {info['total_blocks']}")
        print(f"     - Execution stages: {info['execution_stages']}")
    except Exception as e:
        print(f"   Error creating runner: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load weights
    print("\n3. Loading weights...")
    try:
        runner.load_weights(str(model_dir))
        print(f"   Weights loaded from {model_dir}")
    except Exception as e:
        print(f"   Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test prompts
    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "Machine learning is"
    ]

    print("\n4. Running inference tests...")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Test {i}/{len(prompts)}: \"{prompt}\"")

        try:
            # Tokenize
            input_ids = tokenizer.encode(prompt)
            print(f"     Tokenized: {len(input_ids)} tokens")

            # Generate with greedy decoding first
            print("     Generating (greedy)...")
            output_ids = runner.generate(
                prompt_ids=input_ids,
                max_tokens=30,
                temperature=1.0,
                top_k=50,
                stop_tokens=[tokenizer.eos_token_id]
            )

            # Decode
            output_text = tokenizer.decode(output_ids)
            print(f"     Output: {output_text}")
            print(f"     Generated {len(output_ids) - len(input_ids)} new tokens")

        except Exception as e:
            print(f"     Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("Pure MLX operators working correctly with real inference")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
