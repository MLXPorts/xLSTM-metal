#!/usr/bin/env python
"""
Test inference with real pretrained xLSTM-7B weights.

This tests that:
1. Weights load correctly
2. Forward pass produces reasonable outputs
3. No NaN or Inf values
4. Logits are in reasonable ranges
5. Greedy decoding works
"""

import sys
from pathlib import Path

sys.path.insert(0, '..')

import mlx.core as mx
from xlstm_metal.mlx_jit.models import WiredxLSTM


def test_load_pretrained():
    """Test 1: Load model with pretrained weights."""
    print("\n" + "=" * 60)
    print("TEST 1: Loading pretrained xLSTM-7B model")
    print("=" * 60)

    model_dir = Path("../xlstm_7b_model")
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return None

    try:
        print("Loading model with pretrained weights...")
        model = WiredxLSTM.from_pretrained(model_dir)

        print(f"✓ Model loaded successfully")
        print(f"  - Blocks: {len(model.blocks)}")
        print(f"  - Vocab size: {model.config['vocab_size']}")
        print(f"  - Embedding dim: {model.config['embedding_dim']}")

        # Check weight statistics
        if model.embedding is not None:
            emb_weights = model.embedding.weight
            emb_mean = float(mx.mean(emb_weights))
            emb_std = float(mx.std(emb_weights))
            emb_min = float(mx.min(emb_weights))
            emb_max = float(mx.max(emb_weights))

            print(f"\n  Embedding weights:")
            print(f"    - mean: {emb_mean:.6f}")
            print(f"    - std:  {emb_std:.6f}")
            print(f"    - min:  {emb_min:.6f}")
            print(f"    - max:  {emb_max:.6f}")

        # Check first block weights
        if len(model.blocks) > 0:
            block = model.blocks[0]
            if hasattr(block, 'mlstm_cell'):
                proj = block.mlstm_cell.projection_cell
                q_weights = proj.q_proj.weight
                q_mean = float(mx.mean(q_weights))
                q_std = float(mx.std(q_weights))

                print(f"\n  Block 0 mLSTM Q projection:")
                print(f"    - mean: {q_mean:.6f}")
                print(f"    - std:  {q_std:.6f}")
                print(f"    - shape: {q_weights.shape}")

        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass_pretrained(model):
    """Test 2: Forward pass with pretrained weights."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward pass with pretrained weights")
    print("=" * 60)

    if model is None:
        print("❌ Model not loaded, skipping test")
        return False

    try:
        # Create some test inputs
        # Using reasonable token IDs (within vocab range)
        B, S = 1, 8
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int64)

        print(f"  Input shape: {input_ids.shape}")
        print(f"  Input IDs: {input_ids.tolist()[0]}")

        # Forward pass
        print("\n  Running forward pass...")
        logits = model(input_ids)

        print(f"✓ Forward pass completed")
        print(f"  Logits shape: {logits.shape}")

        # Check for NaN or Inf
        has_nan = bool(mx.any(mx.isnan(logits)))
        has_inf = bool(mx.any(mx.isinf(logits)))

        if has_nan:
            print(f"❌ Logits contain NaN values!")
            return False
        if has_inf:
            print(f"❌ Logits contain Inf values!")
            return False

        print(f"✓ No NaN or Inf values")

        # Check logit statistics
        logits_mean = float(mx.mean(logits))
        logits_std = float(mx.std(logits))
        logits_min = float(mx.min(logits))
        logits_max = float(mx.max(logits))

        print(f"\n  Logit statistics:")
        print(f"    - mean: {logits_mean:.6f}")
        print(f"    - std:  {logits_std:.6f}")
        print(f"    - min:  {logits_min:.6f}")
        print(f"    - max:  {logits_max:.6f}")

        # Check if logits are in reasonable range
        # Typically logits should be in range [-50, 50] or so
        if abs(logits_mean) > 100 or logits_std > 100:
            print(f"⚠️  Warning: Logits may be in unusual range")
        else:
            print(f"✓ Logits in reasonable range")

        # Get predictions for last token
        last_token_logits = logits[0, -1, :]  # [vocab_size]
        top_k = 10

        # Get top-k predictions
        top_k_indices = mx.argpartition(-last_token_logits, top_k)[:top_k]
        top_k_logits = last_token_logits[top_k_indices]

        print(f"\n  Top-{top_k} predicted token IDs for last position:")
        for i, (idx, logit) in enumerate(zip(top_k_indices.tolist(), top_k_logits.tolist())):
            print(f"    {i + 1}. Token {idx}: logit = {logit:.4f}")

        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_greedy_generation(model):
    """Test 3: Greedy token generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Greedy token generation")
    print("=" * 60)

    if model is None:
        print("❌ Model not loaded, skipping test")
        return False

    try:
        # Start with a short prompt
        prompt_tokens = [1, 2, 3, 4]
        max_new_tokens = 5

        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Generating {max_new_tokens} new tokens...")

        # Generate tokens greedily
        generated_tokens = prompt_tokens.copy()
        state = None

        for step in range(max_new_tokens):
            # Prepare input (just last token for autoregressive generation)
            if step == 0:
                # First step: use full prompt
                input_ids = mx.array([prompt_tokens], dtype=mx.int64)
            else:
                # Subsequent steps: use last generated token
                input_ids = mx.array([[generated_tokens[-1]]], dtype=mx.int64)

            # Forward pass with state
            logits, state = model(input_ids, state=state, return_last_states=True)

            # Get logits for last position
            last_logits = logits[0, -1, :]  # [vocab_size]

            # Greedy: pick highest logit
            next_token = int(mx.argmax(last_logits))

            generated_tokens.append(next_token)

            print(f"    Step {step + 1}: Generated token {next_token} (logit: {float(last_logits[next_token]):.4f})")

        print(f"\n✓ Generation completed")
        print(f"  Full sequence: {generated_tokens}")
        print(f"  New tokens: {generated_tokens[len(prompt_tokens):]}")

        # Check that all tokens are valid
        vocab_size = model.config['vocab_size']
        all_valid = all(0 <= t < vocab_size for t in generated_tokens)

        if all_valid:
            print(f"✓ All generated tokens are valid (0 <= token < {vocab_size})")
        else:
            print(f"❌ Some generated tokens are out of vocab range!")
            return False

        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference(model):
    """Test 4: Batch inference."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch inference with multiple sequences")
    print("=" * 60)

    if model is None:
        print("❌ Model not loaded, skipping test")
        return False

    try:
        # Create batch of inputs
        B, S = 3, 6
        input_ids = mx.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]
        ], dtype=mx.int64)

        print(f"  Batch size: {B}")
        print(f"  Sequence length: {S}")
        print(f"  Input shape: {input_ids.shape}")

        # Forward pass
        print("\n  Running batch forward pass...")
        logits = model(input_ids)

        print(f"✓ Batch forward pass completed")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected: ({B}, {S}, {model.config['vocab_size']})")

        # Check shape
        expected_shape = (B, S, model.config['vocab_size'])
        if logits.shape != expected_shape:
            print(f"❌ Shape mismatch!")
            return False

        print(f"✓ Output shape correct")

        # Check each sequence independently
        print(f"\n  Checking individual sequences:")
        for i in range(B):
            seq_logits = logits[i, -1, :]
            top_token = int(mx.argmax(seq_logits))
            top_logit = float(seq_logits[top_token])

            has_nan = bool(mx.any(mx.isnan(seq_logits)))
            has_inf = bool(mx.any(mx.isinf(seq_logits)))

            status = "✓" if not (has_nan or has_inf) else "❌"
            print(f"    Seq {i}: {status} Top token = {top_token} (logit: {top_logit:.4f})")

            if has_nan or has_inf:
                return False

        print(f"\n✓ All sequences processed correctly")
        return True
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """

    :return:
    """
    print("\nTesting Pretrained xLSTM-7B Inference")
    print("=" * 60)

    # Load model once
    model = test_load_pretrained()

    if model is None:
        print("\n❌ Failed to load model, aborting tests")
        return 1

    # Run inference tests
    tests = [
        ("Forward Pass with Pretrained Weights", lambda: test_forward_pass_pretrained(model)),
        ("Greedy Token Generation", lambda: test_greedy_generation(model)),
        ("Batch Inference", lambda: test_batch_inference(model)),
    ]

    results = [("Model Loading", model is not None)]
    for name, test_fn in tests:
        success = test_fn()
        results.append((name, success))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)
    print("\n" + ("✓ All tests passed!" if all_passed else "❌ Some tests failed"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
