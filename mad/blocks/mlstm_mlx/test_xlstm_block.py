#!/usr/bin/env python
"""
Unit tests for complete xLSTM Block
"""

import mlx.core as mx
from xlstm_block import xLSTMBlock, xLSTMBlockConfig


def test_xlstm_block_shapes():
    """Test xLSTM block produces correct output shapes"""
    print("\n" + "=" * 80)
    print("Test 1: xLSTM Block Shape Validation")
    print("=" * 80)

    # Config matching xLSTM-7B
    config = xLSTMBlockConfig(
        embedding_dim=4096,
        num_heads=8,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0,
        ffn_proj_factor=2.671875,
        use_bias=False
    )

    block = xLSTMBlock(config)

    # Test input
    batch_size = 2
    seq_len = 16
    x = mx.random.normal(shape=(batch_size, seq_len, config.embedding_dim))

    print(f"Config:")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  ffn_proj_up_dim: {config.ffn_config.proj_up_dim}")

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    y, state = block(x, state=None)

    print(f"Output shape: {y.shape}")
    print(f"State tuple:")
    c, n, m = state
    print(f"  C: {c.shape}")
    print(f"  n: {n.shape}")
    print(f"  m: {m.shape}")

    # Validate shape
    assert y.shape == (batch_size, seq_len, config.embedding_dim), \
        f"Output shape mismatch: {y.shape}"

    print("\n✓ All shapes correct")
    print()


def test_xlstm_block_forward_pass():
    """Test xLSTM block forward pass executes without errors"""
    print("=" * 80)
    print("Test 2: xLSTM Block Forward Pass")
    print("=" * 80)

    config = xLSTMBlockConfig(
        embedding_dim=512,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        ffn_proj_factor=2.5
    )

    block = xLSTMBlock(config)

    x = mx.random.normal(shape=(1, 8, config.embedding_dim))

    print(f"Input: {x.shape}")

    y, state = block(x)

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


def test_xlstm_block_residuals():
    """Test that residual connections work"""
    print("=" * 80)
    print("Test 3: Residual Connections")
    print("=" * 80)

    config = xLSTMBlockConfig(
        embedding_dim=256,
        num_heads=4,
        ffn_proj_factor=2.0
    )

    block = xLSTMBlock(config)

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

    # Should be different due to mLSTM + FFN processing
    assert diff_mean > 0.01, "Output too similar to input (residuals not working?)"

    print("✓ Residual connections work correctly")
    print()


def test_xlstm_block_state_carryover():
    """Test state carryover across sequences"""
    print("=" * 80)
    print("Test 4: State Carryover")
    print("=" * 80)

    config = xLSTMBlockConfig(
        embedding_dim=256,
        num_heads=4
    )

    block = xLSTMBlock(config)

    # First sequence
    x1 = mx.random.normal(shape=(1, 4, config.embedding_dim))
    y1, state1 = block(x1, state=None)

    print(f"First call:")
    print(f"  Input: {x1.shape}")
    print(f"  Output: {y1.shape}")

    # Second sequence with state
    x2 = mx.random.normal(shape=(1, 4, config.embedding_dim))
    y2, state2 = block(x2, state=state1)

    print(f"\nSecond call (with state):")
    print(f"  Input: {x2.shape}")
    print(f"  Output: {y2.shape}")

    # Verify state shapes match
    assert state2[0].shape == state1[0].shape, "C state shape mismatch"
    assert state2[1].shape == state1[1].shape, "n state shape mismatch"
    assert state2[2].shape == state1[2].shape, "m state shape mismatch"

    print("✓ State carryover works")
    print()


def test_large_scale_config():
    """Test with xLSTM-7B scale configuration"""
    print("=" * 80)
    print("Test 5: Large Scale (xLSTM-7B) Configuration")
    print("=" * 80)

    config = xLSTMBlockConfig(
        embedding_dim=4096,
        num_heads=8,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0,
        ffn_proj_factor=2.671875,
        use_bias=False
    )

    block = xLSTMBlock(config)

    # Smaller batch for memory
    x = mx.random.normal(shape=(1, 4, config.embedding_dim))

    print(f"Config (xLSTM-7B scale):")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  qk_dim: {config.mlstm_config.qk_dim}")
    print(f"  v_dim: {config.mlstm_config.v_dim}")
    print(f"  ffn_proj_up_dim: {config.ffn_config.proj_up_dim}")

    print(f"\nInput: {x.shape}")

    y, state = block(x)

    print(f"Output: {y.shape}")
    print(f"Output mean: {float(mx.mean(y)):.6f}")
    print(f"Output std: {float(mx.std(y)):.6f}")

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(y))
    has_inf = mx.any(mx.isinf(y))

    assert not bool(has_nan), "Output contains NaN"
    assert not bool(has_inf), "Output contains Inf"

    print("✓ Large scale configuration works")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("xLSTM Block Unit Tests")
    print("=" * 80)

    try:
        test_xlstm_block_shapes()
        test_xlstm_block_forward_pass()
        test_xlstm_block_residuals()
        test_xlstm_block_state_carryover()
        test_large_scale_config()

        print("=" * 80)
        print("All xLSTM Block Tests Passed! ✓")
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
