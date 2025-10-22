#!/usr/bin/env python
"""
Test MAD MLX implementation generation after kernel fixes.

This test verifies the fixed mLSTM kernel produces reasonable predictions.
"""

import mlx.core as mx
from pathlib import Path

# MAD implementation
from mad.wiring.mlx import create_xlstm_7b_wiring, WiredMADModel
from mad.utils.safetensors_loader import load_safetensors_into_wired_model


def test_mad_generation():
    """Test generation with fixed MAD implementation."""

    model_path = "model_cache/models--NX-AI--xLSTM-7b/snapshots/9dc507bd0939cf372a4a4f667335651d8e49dddb"

    print("=" * 80)
    print("Loading MAD xLSTM-7B (MLX) with FIXED kernel...")
    print("=" * 80)

    # Create MAD model
    wiring = create_xlstm_7b_wiring(
        embedding_dim=4096,
        num_heads=8,
        num_blocks=32,
        vocab_size=50304,
        output_logit_soft_cap=30.0
    )
    mad_model = WiredMADModel(wiring, 'embedding', 'lm_head', debug=False)

    # Load weights
    load_safetensors_into_wired_model(model_path, mad_model)

    print(f"✓ MAD model loaded with fixed kernel")

    print("\n" + "=" * 80)
    print("Testing generation with fixed kernel...")
    print("=" * 80)

    # Load tokenizer using MAD TokenizerBlock (avoids transformers/PIL issues)
    from mad.blocks.tokenizer import TokenizerBlock, TokenizerConfig
    tokenizer_config = TokenizerConfig(model_path=model_path, vocab_size=50304)
    tokenizer = TokenizerBlock(tokenizer_config)

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the year 2024,",
    ]

    for prompt in test_prompts:
        print(f"\n{'─' * 80}")
        print(f"Prompt: {repr(prompt)}")
        print(f"{'─' * 80}")

        # Tokenize
        tokens = tokenizer.encode(prompt)  # Returns mx.array
        input_ids = [tokenizer.bos_token_id] + tokens.tolist()

        print(f"Input tokens: {input_ids}")
        print(f"Token count: {len(input_ids)}")

        # Forward pass
        mad_input = mx.array([input_ids])
        result = mad_model(mad_input)

        # Handle both tuple and single return
        if isinstance(result, tuple):
            mad_logits = result[0]
        else:
            mad_logits = result

        print(f"Output shape: {mad_logits.shape}")

        # Get top 5 predictions for last token
        last_logits = mad_logits[0, -1, :]
        top5_indices = mx.argsort(last_logits)[::-1][:5]

        print(f"\nTop 5 predictions:")
        for i, idx in enumerate(top5_indices):
            idx_val = int(idx)
            logit_val = float(last_logits[idx])
            token_text = tokenizer.decode([idx_val])
            print(f"  {i+1}. Token {idx_val:5d} ({repr(token_text):20s}): {logit_val:8.4f}")

        # Generate next token greedily
        predicted_token = int(top5_indices[0])
        continuation = tokenizer.decode([predicted_token])
        print(f"\nGreedy prediction: {prompt}{continuation}")

    print("\n" + "=" * 80)
    print("Kernel Test Complete")
    print("=" * 80)
    print("\nExpected behavior:")
    print("- 'The capital of France is' should predict ' Paris' or similar")
    print("- Logits should be in reasonable range (not NaN/Inf)")
    print("- Top predictions should be semantically plausible")


if __name__ == "__main__":
    test_mad_generation()
