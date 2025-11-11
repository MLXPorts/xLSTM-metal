#!/usr/bin/env python
"""
Test sLSTM NCPS-style implementation.

This tests:
1. sLSTMCell creation
2. xLSTMsLSTMCell block wrapper
3. Forward pass with states
4. Auto-wiring integration
"""

import sys

sys.path.insert(0, '.')

import mlx.core as mx
from xlstm_metal.mlx_jit.blocks.slstm import sLSTMNeuron
from xlstm_metal.mlx_jit.models import xLSTMsLSTMCell


def test_slstm_cell():
    """Test 1: sLSTMNeuron creation and forward pass."""
    print("\n" + "=" * 60)
    print("TEST 1: sLSTMNeuron creation and forward pass")
    print("=" * 60)

    try:
        # Create sLSTM neuron
        cell = sLSTMNeuron(
            input_size=512,
            num_heads=4,
            head_dim=128,
            use_bias=False,
            eps=1e-6,
            gate_soft_cap=15.0
        )

        print(f"✓ sLSTMNeuron created")
        print(f"  - input_size: {cell.input_size}")
        print(f"  - num_heads: {cell.num_heads}")
        print(f"  - head_dim: {cell.head_dim}")

        # Create dummy input
        B, S, D = 2, 8, 512
        x = mx.random.normal(shape=(B, S, D), dtype=mx.float32)

        print(f"\n  Input shape: {x.shape}")

        # Forward pass
        output, state = cell(x)

        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({B}, {S}, {D})")

        # Check output shape
        if output.shape != (B, S, D):
            print(f"❌ Output shape mismatch!")
            return False

        print(f"✓ Output shape correct")

        # Check state
        c, n, m = state
        print(f"\n  State shapes:")
        print(f"    - c (cell state): {c.shape}")
        print(f"    - n (normalizer): {n.shape}")
        print(f"    - m (stabilizer): {m.shape}")

        # Check for NaN/Inf
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))

        if has_nan or has_inf:
            print(f"❌ Output contains NaN or Inf!")
            return False

        print(f"✓ No NaN or Inf values")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slstm_block_wrapper():
    """Test 2: xLSTMsLSTMCell block wrapper."""
    print("\n" + "=" * 60)
    print("TEST 2: xLSTMsLSTMCell block wrapper")
    print("=" * 60)

    try:
        # Create block wrapper
        block = xLSTMsLSTMCell(
            block_index=0,
            embedding_dim=1024,
            num_heads=4,
            head_dim=256,
            gate_soft_cap=15.0,
            ffn_proj_factor=2.667,
            norm_eps=1e-6,
            use_bias=False,
            eps=1e-6
        )

        print(f"✓ xLSTMsLSTMCell created")
        print(f"  - block_index: {block.block_index}")
        print(f"  - embedding_dim: {block.embedding_dim}")
        print(f"  - num_heads: {block.num_heads}")
        print(f"  - head_dim: {block.head_dim}")
        print(f"  - ffn_hidden_dim: {block.ffn_hidden_dim}")

        # Create dummy input
        B, S, D = 1, 16, 1024
        x = mx.random.normal(shape=(B, S, D), dtype=mx.float32)

        print(f"\n  Input shape: {x.shape}")

        # Forward pass
        output, state = block(x)

        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({B}, {S}, {D})")

        # Check output shape
        if output.shape != (B, S, D):
            print(f"❌ Output shape mismatch!")
            return False

        print(f"✓ Output shape correct")

        # Check state
        c, n, m = state
        print(f"\n  State shapes:")
        print(f"    - c: {c.shape}")
        print(f"    - n: {n.shape}")
        print(f"    - m: {m.shape}")

        # Check for NaN/Inf
        has_nan = bool(mx.any(mx.isnan(output)))
        has_inf = bool(mx.any(mx.isinf(output)))

        if has_nan or has_inf:
            print(f"❌ Output contains NaN or Inf!")
            return False

        print(f"✓ No NaN or Inf values")

        # Check weight mapping
        weight_keys = block.get_weight_keys()
        print(f"\n  Weight mapping (first 5):")
        for i, (param_path, safetensors_key) in enumerate(list(weight_keys.items())[:5]):
            print(f"    {param_path} -> {safetensors_key}")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slstm_from_config():
    """Test 3: Create sLSTM block from config."""
    print("\n" + "=" * 60)
    print("TEST 3: Create sLSTM block from config")
    print("=" * 60)

    try:
        # Create config
        config = {
            'embedding_dim': 2048,
            'num_heads': 8,
            'head_dim': 256,
            'gate_soft_cap': 15.0,
            'ffn_proj_factor': 2.667,
            'ffn_round_up_to_multiple_of': 64,
            'norm_eps': 1e-6,
            'use_bias': False,
            'eps': 1e-6,
        }

        # Create block from config
        block = xLSTMsLSTMCell.from_config(block_index=5, config=config)

        print(f"✓ Block created from config")
        print(f"  - block_index: {block.block_index}")
        print(f"  - embedding_dim: {block.embedding_dim}")
        print(f"  - num_heads: {block.num_heads}")

        # Test forward pass
        B, S, D = 1, 4, 2048
        x = mx.random.normal(shape=(B, S, D), dtype=mx.float32)

        output, state = block(x)

        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")

        if output.shape != (B, S, D):
            print(f"❌ Output shape mismatch!")
            return False

        print(f"✓ Output shape correct")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stateful_generation():
    """Test 4: Stateful generation with sLSTM."""
    print("\n" + "=" * 60)
    print("TEST 4: Stateful generation with sLSTM")
    print("=" * 60)

    try:
        # Create block
        block = xLSTMsLSTMCell(
            block_index=0,
            embedding_dim=512,
            num_heads=4,
            head_dim=128,
        )

        print(f"✓ Block created for stateful generation")

        # Generate sequence step by step
        B, D = 1, 512
        num_steps = 5

        print(f"\n  Generating {num_steps} steps...")

        state = None
        for step in range(num_steps):
            # Single token input
            x = mx.random.normal(shape=(B, 1, D), dtype=mx.float32)

            # Forward with state
            output, state = block(x, state=state)

            print(f"    Step {step + 1}: output shape = {output.shape}, state types = {len(state)}")

            # Check state persistence
            c, n, m = state
            if c.shape[0] != B or c.shape[1] != 4:  # num_heads=4
                print(f"❌ State shape incorrect at step {step + 1}")
                return False

        print(f"\n✓ Stateful generation successful")
        print(f"  Final state shapes:")
        print(f"    - c: {state[0].shape}")
        print(f"    - n: {state[1].shape}")
        print(f"    - m: {state[2].shape}")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\nTesting sLSTM NCPS Implementation")
    print("=" * 60)

    tests = [
        ("sLSTMNeuron Creation", test_slstm_cell),
        ("xLSTMsLSTMCell Block Wrapper", test_slstm_block_wrapper),
        ("Create from Config", test_slstm_from_config),
        ("Stateful Generation", test_stateful_generation),
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
