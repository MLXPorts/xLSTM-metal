#!/usr/bin/env python
"""
Compare canonical xlstm-pytorch implementation with MAD MLX implementation.

This test loads the official xLSTM-7B model using both:
1. Official xlstm package (PyTorch) - GROUND TRUTH
2. MAD implementation (MLX) - OUR IMPLEMENTATION

Then compares forward pass outputs to identify differences.
"""

import torch
import mlx.core as mx
from pathlib import Path

# Canonical xlstm loading
from xlstm.xlstm_large.from_pretrained import load_from_pretrained

# MAD implementation
from mad.wiring.mlx import create_xlstm_7b_wiring, WiredMADModel
from mad.utils.safetensors_loader import load_safetensors_into_wired_model


def test_forward_pass_comparison():
    """Compare forward pass outputs between canonical and MAD implementations."""

    model_path = "model_cache/models--NX-AI--xLSTM-7b/snapshots/9dc507bd0939cf372a4a4f667335651d8e49dddb"

    print("=" * 80)
    print("Loading CANONICAL xLSTM-7B (PyTorch)...")
    print("=" * 80)

    # Load canonical model
    canonical_model = load_from_pretrained(
        model_path,
        return_last_states=True,
        backend_mode="inference"
    )
    canonical_model.eval()

    print(f"✓ Canonical model loaded")
    print(f"  Config: {canonical_model.config}")

    print("\n" + "=" * 80)
    print("Loading MAD xLSTM-7B (MLX)...")
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

    print(f"✓ MAD model loaded")

    print("\n" + "=" * 80)
    print("Testing forward pass with same input...")
    print("=" * 80)

    # Create test input: [BOS] "The capital of France is"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_prompt = "The capital of France is"
    tokens = tokenizer.encode(test_prompt, add_special_tokens=False)

    # Prepend BOS
    input_ids = [tokenizer.bos_token_id] + tokens

    print(f"Input: {test_prompt}")
    print(f"Tokens: {input_ids}")
    print(f"Token count: {len(input_ids)}")

    # Canonical forward (PyTorch)
    with torch.no_grad():
        canonical_input = torch.tensor([input_ids])
        canonical_logits, _ = canonical_model(canonical_input)

    print(f"\nCanonical output shape: {canonical_logits.shape}")
    print(f"Canonical last token logits (top 5):")
    last_logits = canonical_logits[0, -1, :]
    top5_canonical = torch.topk(last_logits, 5)
    for i, (val, idx) in enumerate(zip(top5_canonical.values, top5_canonical.indices)):
        token_text = tokenizer.decode([idx.item()])
        print(f"  {i+1}. Token {idx.item()} ({repr(token_text)}): {val.item():.4f}")

    # MAD forward (MLX)
    mad_input = mx.array([input_ids])
    mad_logits, _ = mad_model(mad_input, state=None)

    print(f"\nMAD output shape: {mad_logits.shape}")
    print(f"MAD last token logits (top 5):")
    last_logits_mad = mad_logits[0, -1, :]
    top5_indices = mx.argsort(last_logits_mad)[::-1][:5]
    for i, idx in enumerate(top5_indices):
        idx_val = int(idx)
        logit_val = float(last_logits_mad[idx])
        token_text = tokenizer.decode([idx_val])
        print(f"  {i+1}. Token {idx_val} ({repr(token_text)}): {logit_val:.4f}")

    # Compare outputs
    print("\n" + "=" * 80)
    print("Comparison...")
    print("=" * 80)

    # Convert to MLX arrays for comparison
    canonical_mx = mx.array(canonical_logits[0, -1, :].cpu().numpy())
    mad_mx = mad_logits[0, -1, :]

    # Compute differences using MLX
    diff = mad_mx - canonical_mx
    max_diff = float(mx.abs(diff).max())
    mean_diff = float(mx.abs(diff).mean())

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # Check if predictions match
    canonical_pred = int(top5_canonical.indices[0].item())
    mad_pred = int(top5_indices[0])

    print(f"\nCanonical predicts: {canonical_pred} ({repr(tokenizer.decode([canonical_pred]))})")
    print(f"MAD predicts: {mad_pred} ({repr(tokenizer.decode([mad_pred]))})")

    if canonical_pred == mad_pred:
        print("✅ PREDICTIONS MATCH!")
    else:
        print("❌ PREDICTIONS DIFFER - debugging needed")

        # Show where they differ most using MLX
        abs_diff = mx.abs(diff)
        top_diffs = mx.argsort(abs_diff)[::-1][:10]
        print("\nTop 10 largest differences:")
        for i, idx in enumerate(top_diffs):
            idx_val = int(idx)
            print(f"  {i+1}. Token {idx_val} ({repr(tokenizer.decode([idx_val]))}): "
                  f"canonical={float(canonical_mx[idx_val]):.4f}, mad={float(mad_mx[idx_val]):.4f}, "
                  f"diff={float(diff[idx_val]):.4f}")


if __name__ == "__main__":
    test_forward_pass_comparison()
