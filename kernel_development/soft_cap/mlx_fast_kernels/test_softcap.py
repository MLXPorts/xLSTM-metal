import mlx.core as mx
import numpy as np
from softcap import soft_cap

def test_softcap():
    print("=" * 60)
    print("Soft Cap Kernel Correctness Tests")
    print("=" * 60)

    # Test 1: Zero input
    print("\nTest 1: Zero input")
    x1 = mx.array([0.0], dtype=mx.float32)
    y1 = soft_cap(x1, 5.0)
    assert abs(y1[0].item() - 0.0) < 1e-5
    print(f"  Input: {x1[0].item():.4f}, Output: {y1[0].item():.4f}, Expected: 0.0")
    print("  ✓ Passed")

    # Test 2: Input equals cap (tanh(1) ≈ 0.7616)
    print("\nTest 2: Input equals cap (x=5, cap=5)")
    x2 = mx.array([5.0], dtype=mx.float32)
    y2 = soft_cap(x2, 5.0)
    assert abs(y2[0].item() - 3.808) < 0.01
    print(f"  Input: {x2[0].item():.4f}, Output: {y2[0].item():.4f}, Expected: ~3.808")
    print("  ✓ Passed")

    # Test 3: Large positive input (saturates near cap)
    print("\nTest 3: Large positive input (saturation)")
    x3 = mx.array([50.0], dtype=mx.float32)
    y3 = soft_cap(x3, 5.0)
    assert abs(y3[0].item() - 5.0) < 0.01
    print(f"  Input: {x3[0].item():.4f}, Output: {y3[0].item():.4f}, Expected: ~5.0")
    print("  ✓ Passed")

    # Test 4: Large negative input (saturates near -cap)
    print("\nTest 4: Large negative input (saturation)")
    x4 = mx.array([-50.0], dtype=mx.float32)
    y4 = soft_cap(x4, 5.0)
    assert abs(y4[0].item() - (-5.0)) < 0.01
    print(f"  Input: {x4[0].item():.4f}, Output: {y4[0].item():.4f}, Expected: ~-5.0")
    print("  ✓ Passed")

    # Test 5: Small input (linear region, output ≈ input)
    print("\nTest 5: Small input (linear region)")
    x5 = mx.array([0.1], dtype=mx.float32)
    y5 = soft_cap(x5, 5.0)
    assert abs(y5[0].item() - 0.1) < 0.01
    print(f"  Input: {x5[0].item():.4f}, Output: {y5[0].item():.4f}, Expected: ~0.1")
    print("  ✓ Passed")

    # Test 6: Different cap value
    print("\nTest 6: Different cap value (cap=2)")
    x6 = mx.array([2.0], dtype=mx.float32)
    y6 = soft_cap(x6, 2.0)
    assert abs(y6[0].item() - 1.523) < 0.01
    print(f"  Input: {x6[0].item():.4f}, Output: {y6[0].item():.4f}, Expected: ~1.523")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_softcap()
