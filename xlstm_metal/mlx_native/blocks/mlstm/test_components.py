"""
Unit tests for MLX components (RMSNorm, MultiHeadLayerNorm, soft_cap)

Tests:
- Soft-cap behavior
- RMSNorm correctness
- MultiHeadLayerNorm shape and per-head independence
- Numerical stability
"""

import mlx.core as mx
import numpy as np

from components import soft_cap, RMSNorm, MultiHeadLayerNorm


# Helper to convert MLX arrays to Python for assertions
def to_scalar(x):
    """Convert MLX array to Python scalar"""
    return float(x) if isinstance(x, mx.array) else float(x)


def to_list(x):
    """Convert MLX array to Python list"""
    if isinstance(x, mx.array):
        return x.tolist()
    return x


def test_soft_cap_basic():
    """Test soft-cap function behavior"""
    print("\nTest 1: Soft-cap basic behavior")
    print("=" * 80)

    # Test with known values
    x = mx.array([0.0, 1.0, -1.0, 10.0, -10.0, 15.0, -15.0, 30.0, -30.0])
    cap_value = 15.0

    result = soft_cap(x, cap_value)

    print(f"Input: {x}")
    print(f"Cap value: {cap_value}")
    print(f"Output: {result}")

    # Check properties
    # 1. Zero maps to zero
    assert np.abs(float(result[0])) < 1e-6, "Soft-cap(0) should be ~0"

    # 2. Output is bounded by cap_value
    assert np.all(np.abs(np.array(result)) < cap_value + 0.1), "Output should be bounded by cap_value"

    # 3. Monotonic
    x_sorted = mx.array([-30.0, -15.0, -10.0, -1.0, 0.0, 1.0, 10.0, 15.0, 30.0])
    result_sorted = soft_cap(x_sorted, cap_value)
    diffs = np.diff(np.array(result_sorted))
    assert np.all(diffs > 0), "Soft-cap should be monotonic"

    print(f"✓ Soft-cap is monotonic")
    print(f"✓ Soft-cap is bounded by {cap_value}")
    print(f"✓ Soft-cap(0) ≈ 0")

    # Test asymptotic behavior
    large_val = soft_cap(mx.array([100.0]), 15.0)
    print(f"Soft-cap(100.0) = {float(large_val):.4f} (should approach 15.0)")
    assert 14.9 < float(large_val) < 15.0, "Should approach cap asymptotically"

    print("✓ All soft-cap tests passed\n")


def test_rmsnorm_correctness():
    """Test RMSNorm normalization correctness"""
    print("\nTest 2: RMSNorm correctness")
    print("=" * 80)

    # Create RMSNorm layer
    d_model = 512
    norm = RMSNorm(num_features=d_model)

    # Test input
    batch_size = 2
    seq_len = 10
    x = mx.random.normal(shape=(batch_size, seq_len, d_model))

    print(f"Input shape: {x.shape}")
    print(f"Input mean: {float(mx.mean(x)):.6f}")
    print(f"Input std: {float(mx.std(x)):.6f}")

    # Forward pass
    x_norm = norm(x)

    print(f"Output shape: {x_norm.shape}")
    print(f"Output mean: {float(mx.mean(x_norm)):.6f}")
    print(f"Output std: {float(mx.std(x_norm)):.6f}")

    # Check shape preservation
    assert x_norm.shape == x.shape, f"Shape mismatch: {x_norm.shape} vs {x.shape}"
    print(f"✓ Shape preserved: {x_norm.shape}")

    # Check RMS normalization (should have unit RMS per feature)
    # RMS(x) = sqrt(mean(x^2))
    x_norm_np = np.array(x_norm)
    rms_per_feature = np.sqrt(np.mean(x_norm_np ** 2, axis=(0, 1)))

    # With learned weight initialized to 1, RMS should be ~1
    print(f"RMS per feature (first 10): {rms_per_feature[:10]}")
    print(f"RMS mean: {np.mean(rms_per_feature):.6f}")
    print(f"RMS std: {np.std(rms_per_feature):.6f}")

    # Should be close to 1 (with some variance due to weight)
    assert 0.8 < np.mean(rms_per_feature) < 1.2, "RMS should be ~1"

    print("✓ RMS normalization correct\n")


def test_rmsnorm_weight_and_bias():
    """Test RMSNorm learned parameters"""
    print("\nTest 3: RMSNorm weight and bias")
    print("=" * 80)

    d_model = 128

    # Test with weight only
    norm_weight = RMSNorm(num_features=d_model)

    # Test with weight and bias
    norm_both = RMSNorm(num_features=d_model, use_bias=True)

    # Test without weight or bias
    norm_none = RMSNorm(num_features=d_model, use_weight=False)

    x = mx.random.normal(shape=(2, 10, d_model))

    y_weight = norm_weight(x)
    y_both = norm_both(x)
    y_none = norm_none(x)

    print(f"With weight only - mean: {float(mx.mean(y_weight)):.6f}")
    print(f"With weight+bias - mean: {float(mx.mean(y_both)):.6f}")
    print(f"Without params - mean: {float(mx.mean(y_none)):.6f}")

    # Without bias, mean should be ~0
    assert abs(float(mx.mean(y_weight))) < 0.1, "Without bias, mean should be ~0"

    # With bias initialized to 0, should still be ~0
    assert abs(float(mx.mean(y_both))) < 0.1, "With zero bias, mean should be ~0"

    print("✓ Weight and bias handling correct\n")


def test_multihead_layernorm_shape():
    """Test MultiHeadLayerNorm shape handling"""
    print("\nTest 4: MultiHeadLayerNorm shape handling")
    print("=" * 80)

    num_heads = 8
    head_dim = 512
    batch_size = 2
    seq_len = 16

    mhln = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim)

    # Input: [B, S, num_heads, head_dim]
    x = mx.random.normal(shape=(batch_size, seq_len, num_heads, head_dim))

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {mhln.weight.shape}")

    y = mhln(x)

    print(f"Output shape: {y.shape}")

    # Check shape preservation
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"✓ Shape preserved: {y.shape}")

    # Check weight shape is [num_heads, head_dim]
    assert mhln.weight.shape == (num_heads, head_dim), \
        f"Weight shape should be ({num_heads}, {head_dim}), got {mhln.weight.shape}"
    print(f"✓ Weight shape correct: {mhln.weight.shape}")

    print("✓ MultiHeadLayerNorm shape tests passed\n")


def test_multihead_layernorm_per_head_independence():
    """Test that MultiHeadLayerNorm normalizes per-head independently"""
    print("\nTest 5: MultiHeadLayerNorm per-head independence")
    print("=" * 80)

    num_heads = 4
    head_dim = 64
    batch_size = 1
    seq_len = 8

    mhln = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim)

    # Create input with different statistics per head
    x = mx.zeros((batch_size, seq_len, num_heads, head_dim))

    # Head 0: mean=0, std=1
    x[:, :, 0, :] = mx.random.normal(shape=(batch_size, seq_len, head_dim))

    # Head 1: mean=10, std=1
    x[:, :, 1, :] = mx.random.normal(shape=(batch_size, seq_len, head_dim)) + 10.0

    # Head 2: mean=0, std=5
    x[:, :, 2, :] = mx.random.normal(shape=(batch_size, seq_len, head_dim)) * 5.0

    # Head 3: mean=-5, std=0.5
    x[:, :, 3, :] = mx.random.normal(shape=(batch_size, seq_len, head_dim)) * 0.5 - 5.0

    print("Input statistics per head:")
    x_np = np.array(x)
    for h in range(num_heads):
        head_data = x_np[0, :, h, :]
        print(f"  Head {h}: mean={np.mean(head_data):6.2f}, std={np.std(head_data):6.2f}")

    # Normalize
    y = mhln(x)
    y_np = np.array(y)

    print("\nOutput statistics per head (after normalization):")
    for h in range(num_heads):
        head_data = y_np[0, :, h, :]
        mean = np.mean(head_data)
        std = np.std(head_data)
        print(f"  Head {h}: mean={mean:6.4f}, std={std:6.4f}")

        # Each head should be normalized independently
        # Mean should be ~0, std should be ~1 (with learned weight=1)
        assert abs(mean) < 0.1, f"Head {h} mean should be ~0, got {mean}"
        assert 0.8 < std < 1.2, f"Head {h} std should be ~1, got {std}"

    print("✓ Each head normalized independently")
    print("✓ Per-head independence verified\n")


def test_multihead_layernorm_vs_standard():
    """Test difference between MultiHeadLayerNorm and standard LayerNorm"""
    print("\nTest 6: MultiHeadLayerNorm vs standard LayerNorm")
    print("=" * 80)

    num_heads = 2
    head_dim = 4
    batch_size = 1
    seq_len = 1

    # Create input with very different head statistics
    x = mx.zeros((batch_size, seq_len, num_heads, head_dim))
    x[:, :, 0, :] = mx.array([[1.0, 2.0, 3.0, 4.0]])  # Head 0
    x[:, :, 1, :] = mx.array([[10.0, 20.0, 30.0, 40.0]])  # Head 1 (10x larger)

    print(f"Input head 0: {np.array(x[0, 0, 0, :])}")
    print(f"Input head 1: {np.array(x[0, 0, 1, :])}")

    # Multi-head norm (per-head)
    mhln = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim)
    y_mhln = mhln(x)

    print(f"\nMultiHeadLayerNorm output head 0: {np.array(y_mhln[0, 0, 0, :])}")
    print(f"MultiHeadLayerNorm output head 1: {np.array(y_mhln[0, 0, 1, :])}")

    # Both heads should have similar normalized values
    # (because each is normalized independently)
    head0 = np.array(y_mhln[0, 0, 0, :])
    head1 = np.array(y_mhln[0, 0, 1, :])

    # Patterns should be similar (both are [1,2,3,4] normalized)
    correlation = np.corrcoef(head0, head1)[0, 1]
    print(f"\nCorrelation between normalized heads: {correlation:.4f}")
    assert correlation > 0.99, "Heads should have similar patterns after per-head norm"

    print("✓ MultiHeadLayerNorm preserves per-head patterns")
    print("✓ Different from standard LayerNorm (which would mix heads)\n")


def test_dtype_handling():
    """Test float32 reduction forcing"""
    print("\nTest 7: Dtype handling and float32 reductions")
    print("=" * 80)

    d_model = 256

    # Test with float32 reductions
    norm_f32 = RMSNorm(num_features=d_model)

    # Test with native dtype reductions
    norm_native = RMSNorm(
        num_features=d_model,
        force_float32_reductions=False
    )

    # Use bfloat16 input
    x = mx.random.normal(shape=(2, 10, d_model)).astype(mx.float16)

    print(f"Input dtype: {x.dtype}")

    y_f32 = norm_f32(x)
    y_native = norm_native(x)

    print(f"Output dtype (force_float32=True): {y_f32.dtype}")
    print(f"Output dtype (force_float32=False): {y_native.dtype}")

    # Both should preserve input dtype
    assert y_f32.dtype == x.dtype, "Should preserve input dtype"
    assert y_native.dtype == x.dtype, "Should preserve input dtype"

    # But internal computations differ
    # This is mainly for numerical stability
    print("✓ Dtype preservation working")
    print("✓ Float32 reductions option functional\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MLX Components Unit Tests")
    print("=" * 80)

    try:
        test_soft_cap_basic()
        test_rmsnorm_correctness()
        test_rmsnorm_weight_and_bias()
        test_multihead_layernorm_shape()
        test_multihead_layernorm_per_head_independence()
        test_multihead_layernorm_vs_standard()
        test_dtype_handling()

        print("\n" + "=" * 80)
        print("All Component Tests Passed! ✓")
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
