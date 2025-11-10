#!/usr/bin/env python
"""
Test WiredxLSTM model creation and inference.

This tests the NCPS-style model wrapper that automatically builds
the correct architecture from safetensors structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, '.')

import mlx.core as mx
from xlstm_metal.mlx_jit.models import WiredxLSTM


def test_model_creation():
    """Test 1: Can we create a WiredxLSTM model?"""
    print("\n" + "=" * 60)
    print("TEST 1: Creating WiredxLSTM from safetensors")
    print("=" * 60)

    model_dir = Path("xlstm_7b_model")
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False

    try:
        # Create model without loading weights (faster)
        model = WiredxLSTM.from_pretrained(
            model_dir,
            load_weights=False
        )

        print(f"✓ Model created successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Number of blocks: {len(model.blocks)}")
        print(f"  - Block types: {set(model.wiring.block_types.values())}")
        print(f"  - Has embedding: {model.embedding is not None}")
        print(f"  - Has out_norm: {model.out_norm is not None}")
        print(f"  - Has lm_head: {model.lm_head is not None}")

        # Check config
        config = model.get_config()
        print(f"\n  Config:")
        print(f"    - embedding_dim: {config['embedding_dim']}")
        print(f"    - vocab_size: {config['vocab_size']}")
        print(f"    - num_blocks: {config['num_blocks']}")

        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test 2: Can we run a forward pass?"""
    print("\n" + "=" * 60)
    print("TEST 2: Running forward pass")
    print("=" * 60)

    try:
        model = WiredxLSTM.from_pretrained(
            "xlstm_7b_model",
            load_weights=False
        )

        # Create dummy input
        B, S = 1, 4
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        print(f"  Input shape: {input_ids.shape}")

        # Forward pass
        logits = model(input_ids)

        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected shape: ({B}, {S}, {model.config['vocab_size']})")

        # Verify shape
        expected_shape = (B, S, model.config['vocab_size'])
        if logits.shape == expected_shape:
            print(f"✓ Output shape correct")
        else:
            print(f"❌ Output shape incorrect: got {logits.shape}, expected {expected_shape}")
            return False

        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_with_states():
    """Test 3: Can we run forward pass with state management?"""
    print("\n" + "=" * 60)
    print("TEST 3: Running forward pass with state management")
    print("=" * 60)

    try:
        model = WiredxLSTM.from_pretrained(
            "xlstm_7b_model",
            load_weights=False
        )

        # Create dummy input
        B, S = 1, 4
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        # Forward pass with state return
        logits, states = model(input_ids, return_last_states=True)

        print(f"✓ Forward pass with states successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Number of block states: {len(states)}")
        print(f"  Expected states: {len(model.blocks)}")

        # Verify state count
        if len(states) == len(model.blocks):
            print(f"✓ State count correct")
        else:
            print(f"❌ State count incorrect")
            return False

        # Check first block state structure
        if states[0] is not None:
            print(f"\n  Block 0 state:")
            print(f"    - C shape: {states[0][0].shape}")
            print(f"    - n shape: {states[0][1].shape}")
            print(f"    - m shape: {states[0][2].shape}")

        return True
    except Exception as e:
        print(f"❌ Forward pass with states failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_loading():
    """Test 4: Can we load pretrained weights?"""
    print("\n" + "=" * 60)
    print("TEST 4: Loading pretrained weights")
    print("=" * 60)

    try:
        # Create model with weight loading
        print("  Creating model and loading weights...")
        model = WiredxLSTM.from_pretrained(
            "xlstm_7b_model",
            load_weights=True
        )

        print(f"✓ Model created with weights loaded")

        # Verify weights are not all zeros
        if model.embedding is not None:
            emb_mean = mx.mean(mx.abs(model.embedding.weight))
            print(f"  Embedding weight |mean|: {float(emb_mean):.6f}")
            if float(emb_mean) > 0:
                print(f"✓ Embedding weights loaded (non-zero)")
            else:
                print(f"❌ Embedding weights appear to be zeros")
                return False

        # Check first block weights
        if len(model.blocks) > 0:
            block = model.blocks[0]
            if hasattr(block, 'norm_mlstm'):
                norm_mean = mx.mean(mx.abs(block.norm_mlstm.weight))
                print(f"  Block 0 norm weight |mean|: {float(norm_mean):.6f}")
                if float(norm_mean) > 0:
                    print(f"✓ Block weights loaded (non-zero)")
                else:
                    print(f"❌ Block weights appear to be zeros")
                    return False

        return True
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\nTesting WiredxLSTM Model")
    print("=" * 60)

    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Forward Pass with States", test_forward_pass_with_states),
        ("Weight Loading", test_weight_loading),
    ]

    results = []
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
