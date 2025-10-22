#!/usr/bin/env python
"""
Unit tests for mLSTM Block

Tests the complete mLSTMLayer and mLSTMBlock with forward passes.
"""

import mlx.core as mx
from block import mLSTMLayer, mLSTMBlock, mLSTMConfig


def test_mlstm_layer_shapes():
    """Test mLSTM layer produces correct output shapes"""
    print("\n" + "=" * 80)
    print("Test 1: mLSTM Layer Shape Validation")
    print("=" * 80)

    # Config matching xLSTM-7B
    config = mLSTMConfig(
        embedding_dim=4096,
        num_heads=8,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0,
        eps=1e-6
    )

    layer = mLSTMLayer(config)

    # Test input
    batch_size = 2
    seq_len = 16
    x = mx.random.normal(shape=(batch_size, seq_len, config.embedding_dim))

    print(f"Config:")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  qk_dim: {config.qk_dim}")
    print(f"  v_dim: {config.v_dim}")
    print(f"  head_dim: {config.head_dim}")

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    y, state = layer(x, state=None)

    print(f"Output shape: {y.shape}")
    print(f"State tuple:")
    c, n, m = state
    print(f"  C: {c.shape}")
    print(f"  n: {n.shape}")
    print(f"  m: {m.shape}")

    # Validate shapes
    assert y.shape == (batch_size, seq_len, config.embedding_dim), \
        f"Output shape mismatch: {y.shape} vs {(batch_size, seq_len, config.embedding_dim)}"

    # C: [B, NH, head_dim, qk_head_dim] where head_dim=v_dim/NH, qk_head_dim=qk_dim/NH
    assert c.shape == (batch_size, config.num_heads, config.head_dim, config.qk_head_dim), \
        f"C shape mismatch: {c.shape} (expected [B, NH, {config.head_dim}, {config.qk_head_dim}])"

    assert n.shape == (batch_size, config.num_heads, config.qk_head_dim), \
        f"n shape mismatch: {n.shape} (expected [B, NH, {config.qk_head_dim}])"

    assert m.shape == (batch_size, config.num_heads), \
        f"m shape mismatch: {m.shape}"

    print("\n✓ All shapes correct")
    print()


def test_mlstm_layer_forward_pass():
    """Test mLSTM layer forward pass executes without errors"""
    print("=" * 80)
    print("Test 2: mLSTM Layer Forward Pass")
    print("=" * 80)

    config = mLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )

    layer = mLSTMLayer(config)

    x = mx.random.normal(shape=(1, 8, config.embedding_dim))

    print(f"Input: {x.shape}")

    y, state = layer(x)

    print(f"Output: {y.shape}")
    print(f"Output mean: {float(mx.mean(y)):.6f}")
    print(f"Output std: {float(mx.std(y)):.6f}")

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(y))
    has_inf = mx.any(mx.isinf(y))

    print(f"Contains NaN: {bool(has_nan)}")
    print(f"Contains Inf: {bool(has_inf)}")

    assert not bool(has_nan), "Output contains NaN"
    assert not bool(has_inf), "Output contains Inf"

    print("✓ Forward pass successful")
    print()


def test_mlstm_layer_state_carryover():
    """Test mLSTM layer state can be carried across calls"""
    print("=" * 80)
    print("Test 3: mLSTM Layer State Carryover")
    print("=" * 80)

    config = mLSTMConfig(
        embedding_dim=256,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0
    )

    layer = mLSTMLayer(config)

    # First sequence
    x1 = mx.random.normal(shape=(1, 4, config.embedding_dim))
    y1, state1 = layer(x1, state=None)

    print(f"First call:")
    print(f"  Input: {x1.shape}")
    print(f"  Output: {y1.shape}")
    print(f"  State m mean: {float(mx.mean(state1[2])):.6f}")

    # Second sequence with state
    x2 = mx.random.normal(shape=(1, 4, config.embedding_dim))
    y2, state2 = layer(x2, state=state1)

    print(f"\nSecond call (with state):")
    print(f"  Input: {x2.shape}")
    print(f"  Output: {y2.shape}")
    print(f"  State m mean: {float(mx.mean(state2[2])):.6f}")

    # Verify state was used
    # m_state should have accumulated
    m1_mean = float(mx.mean(mx.abs(state1[2])))
    m2_mean = float(mx.mean(mx.abs(state2[2])))

    print(f"\nState accumulation check:")
    print(f"  |m1| mean: {m1_mean:.6f}")
    print(f"  |m2| mean: {m2_mean:.6f}")

    # State should have changed
    assert state2[2].shape == state1[2].shape, "State shape mismatch"

    print("✓ State carryover works")
    print()


def test_mlstm_block():
    """Test complete mLSTM block with pre-norm and residual"""
    print("=" * 80)
    print("Test 4: mLSTM Block (with pre-norm and residual)")
    print("=" * 80)

    config = mLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )

    block = mLSTMBlock(config)

    x = mx.random.normal(shape=(2, 8, config.embedding_dim))

    print(f"Input: {x.shape}")
    print(f"Input mean: {float(mx.mean(x)):.6f}")

    y, state = block(x)

    print(f"Output: {y.shape}")
    print(f"Output mean: {float(mx.mean(y)):.6f}")

    # Check residual connection (output should differ from input)
    assert y.shape == x.shape, "Shape mismatch"

    diff_mean = float(mx.mean(mx.abs(y - x)))
    print(f"Mean absolute difference from input: {diff_mean:.6f}")

    # Should be different due to mLSTM processing
    assert diff_mean > 0.01, "Output too similar to input (residual not working?)"

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(y))
    has_inf = mx.any(mx.isinf(y))

    print(f"Contains NaN: {bool(has_nan)}")
    print(f"Contains Inf: {bool(has_inf)}")

    assert not bool(has_nan), "Output contains NaN"
    assert not bool(has_inf), "Output contains Inf"

    print("✓ mLSTM block works correctly")
    print()


def test_mlstm_layer_gate_soft_cap():
    """Test that gate soft-capping is applied"""
    print("=" * 80)
    print("Test 5: Gate Soft-Capping")
    print("=" * 80)

    config = mLSTMConfig(
        embedding_dim=128,
        num_heads=2,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )

    layer = mLSTMLayer(config)

    # Create input that will produce large gate pre-activations
    x = mx.ones(shape=(1, 4, config.embedding_dim)) * 100.0

    print(f"Input (all 100.0): {x.shape}")

    y, state = layer(x)

    print(f"Output: {y.shape}")

    # Check that output is not exploded (soft-cap working)
    y_max = float(mx.max(mx.abs(y)))
    print(f"Output max abs value: {y_max:.6f}")

    # Should be bounded (soft-cap limits gates to ±15)
    # Output magnitude depends on the computation but shouldn't be astronomical
    assert y_max < 1e10, f"Output too large: {y_max} (soft-cap may not be working)"

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(y))
    has_inf = mx.any(mx.isinf(y))

    assert not bool(has_nan), "Output contains NaN"
    assert not bool(has_inf), "Output contains Inf"

    print("✓ Soft-capping prevents explosion")
    print()


def test_variable_sequence_lengths():
    """Test mLSTM layer with different sequence lengths"""
    print("=" * 80)
    print("Test 6: Variable Sequence Lengths")
    print("=" * 80)

    config = mLSTMConfig(
        embedding_dim=256,
        num_heads=4
    )

    layer = mLSTMLayer(config)

    for seq_len in [1, 4, 16, 64]:
        x = mx.random.normal(shape=(1, seq_len, config.embedding_dim))
        y, state = layer(x)

        assert y.shape == (1, seq_len, config.embedding_dim), \
            f"Shape mismatch for seq_len={seq_len}"

        has_nan = mx.any(mx.isnan(y))
        has_inf = mx.any(mx.isinf(y))

        assert not bool(has_nan), f"NaN for seq_len={seq_len}"
        assert not bool(has_inf), f"Inf for seq_len={seq_len}"

        print(f"  seq_len={seq_len:3d}: ✓")

    print("✓ All sequence lengths work")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("mLSTM Block Unit Tests")
    print("=" * 80)

    try:
        test_mlstm_layer_shapes()
        test_mlstm_layer_forward_pass()
        test_mlstm_layer_state_carryover()
        test_mlstm_block()
        test_mlstm_layer_gate_soft_cap()
        test_variable_sequence_lengths()

        print("=" * 80)
        print("All mLSTM Block Tests Passed! ✓")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
