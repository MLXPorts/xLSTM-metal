"""
Unit tests for Gated FFN Block
"""

import mlx.core as mx

from ffn import GatedFFN, FFNBlock, FFNConfig


def test_gated_ffn_shapes():
    """Test GatedFFN produces correct output shapes"""
    print("\n" + "=" * 80)
    print("Test 1: GatedFFN Shape Validation")
    print("=" * 80)

    # Config matching xLSTM-7B
    config = FFNConfig(
        embedding_dim=4096,
        proj_factor=2.671875,  # proj_up_dim = 10944
        act_fn="swish",
        use_bias=False
    )

    ffn = GatedFFN(config)

    # Test input
    batch_size = 2
    seq_len = 16
    x = mx.random.normal(shape=(batch_size, seq_len, config.embedding_dim))

    print(f"Config:")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  proj_up_dim: {config.proj_up_dim}")
    print(f"  proj_factor: {config.proj_factor}")
    print(f"  act_fn: {config.act_fn}")

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    y = ffn(x)

    print(f"Output shape: {y.shape}")

    # Validate shape
    assert y.shape == (batch_size, seq_len, config.embedding_dim), \
        f"Output shape mismatch: {y.shape} vs {(batch_size, seq_len, config.embedding_dim)}"

    print("✓ Output shape correct")
    print()


def test_gated_ffn_forward_pass():
    """Test GatedFFN forward pass executes without errors"""
    print("=" * 80)
    print("Test 2: GatedFFN Forward Pass")
    print("=" * 80)

    config = FFNConfig(
        embedding_dim=512,
        proj_factor=2.5,
        act_fn="swish"
    )

    ffn = GatedFFN(config)

    x = mx.random.normal(shape=(1, 8, config.embedding_dim))

    print(f"Input: {x.shape}")

    y = ffn(x)

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


def test_activation_functions():
    """Test different activation functions"""
    print("=" * 80)
    print("Test 3: Activation Functions")
    print("=" * 80)

    config_base = FFNConfig(
        embedding_dim=256,
        proj_factor=2.0
    )

    x = mx.random.normal(shape=(1, 4, config_base.embedding_dim))

    for act_fn in ["gelu", "swish", "relu"]:
        config = FFNConfig(
            embedding_dim=config_base.embedding_dim,
            proj_factor=config_base.proj_factor,
            act_fn=act_fn
        )

        ffn = GatedFFN(config)
        y = ffn(x)

        has_nan = mx.any(mx.isnan(y))
        has_inf = mx.any(mx.isinf(y))

        assert not bool(has_nan), f"NaN with {act_fn}"
        assert not bool(has_inf), f"Inf with {act_fn}"

        print(f"  {act_fn:8s}: ✓ (output mean: {float(mx.mean(y)):7.4f})")

    print("✓ All activation functions work")
    print()


def test_gating_behavior():
    """Test that gating mechanism works correctly"""
    print("=" * 80)
    print("Test 4: Gating Mechanism")
    print("=" * 80)

    config = FFNConfig(
        embedding_dim=128,
        proj_factor=2.0,
        act_fn="swish"
    )

    ffn = GatedFFN(config)

    # Test with zeros (should produce near-zero output due to gating)
    x_zeros = mx.zeros((1, 4, config.embedding_dim))
    y_zeros = ffn(x_zeros)

    print(f"Input (zeros): max abs value = {float(mx.max(mx.abs(x_zeros))):.6f}")
    print(f"Output: max abs value = {float(mx.max(mx.abs(y_zeros))):.6f}")

    # Output should be very small (close to zero)
    assert float(mx.max(mx.abs(y_zeros))) < 0.1, "Zero input should produce near-zero output"

    # Test with non-zero input
    x_nonzero = mx.random.normal(shape=(1, 4, config.embedding_dim))
    y_nonzero = ffn(x_nonzero)

    print(f"\nInput (random): mean abs value = {float(mx.mean(mx.abs(x_nonzero))):.6f}")
    print(f"Output: mean abs value = {float(mx.mean(mx.abs(y_nonzero))):.6f}")

    # Output should have non-trivial values
    assert float(mx.mean(mx.abs(y_nonzero))) > 0.01, "Non-zero input should produce non-trivial output"

    print("✓ Gating mechanism works correctly")
    print()


def test_ffn_block():
    """Test complete FFN block with pre-norm and residual"""
    print("=" * 80)
    print("Test 5: FFN Block (with pre-norm and residual)")
    print("=" * 80)

    config = FFNConfig(
        embedding_dim=512,
        proj_factor=2.5,
        act_fn="swish"
    )

    block = FFNBlock(config)

    x = mx.random.normal(shape=(2, 8, config.embedding_dim))

    print(f"Input: {x.shape}")
    print(f"Input mean: {float(mx.mean(x)):.6f}")

    y = block(x)

    print(f"Output: {y.shape}")
    print(f"Output mean: {float(mx.mean(y)):.6f}")

    # Check residual connection
    assert y.shape == x.shape, "Shape mismatch"

    diff_mean = float(mx.mean(mx.abs(y - x)))
    print(f"Mean absolute difference from input: {diff_mean:.6f}")

    # Should be different due to FFN processing
    assert diff_mean > 0.01, "Output too similar to input (residual not working?)"

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(y))
    has_inf = mx.any(mx.isinf(y))

    print(f"Contains NaN: {bool(has_nan)}")
    print(f"Contains Inf: {bool(has_inf)}")

    assert not bool(has_nan), "Output contains NaN"
    assert not bool(has_inf), "Output contains Inf"

    print("✓ FFN block works correctly")
    print()


def test_large_scale_config():
    """Test with xLSTM-7B scale configuration"""
    print("=" * 80)
    print("Test 6: Large Scale (xLSTM-7B) Configuration")
    print("=" * 80)

    config = FFNConfig(
        embedding_dim=4096,
        proj_factor=2.671875,  # proj_up_dim = 10944
        act_fn="swish",
        use_bias=False
    )

    block = FFNBlock(config)

    # Smaller batch for memory
    x = mx.random.normal(shape=(1, 4, config.embedding_dim))

    print(f"Config (xLSTM-7B scale):")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  proj_up_dim: {config.proj_up_dim}")
    print(
        f"  Total FFN params: ~{(config.embedding_dim * 2 * config.proj_up_dim + config.proj_up_dim * config.embedding_dim) / 1e6:.1f}M")

    print(f"\nInput: {x.shape}")

    y = block(x)

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


def test_variable_sequence_lengths():
    """Test FFN with different sequence lengths"""
    print("=" * 80)
    print("Test 7: Variable Sequence Lengths")
    print("=" * 80)

    config = FFNConfig(
        embedding_dim=256,
        proj_factor=2.0
    )

    ffn = GatedFFN(config)

    for seq_len in [1, 4, 16, 64, 128]:
        x = mx.random.normal(shape=(1, seq_len, config.embedding_dim))
        y = ffn(x)

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
    print("Gated FFN Block Unit Tests")
    print("=" * 80)

    try:
        test_gated_ffn_shapes()
        test_gated_ffn_forward_pass()
        test_activation_functions()
        test_gating_behavior()
        test_ffn_block()
        test_large_scale_config()
        test_variable_sequence_lengths()

        print("=" * 80)
        print("All FFN Tests Passed! ✓")
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
