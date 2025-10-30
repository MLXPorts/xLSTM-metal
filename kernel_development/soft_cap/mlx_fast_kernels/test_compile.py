"""
Test to verify the .compile() method works for early compilation.
"""
import mlx.core as mx
import numpy as np
from softcap import SoftCapMLXFastKernel

def test_early_compilation():
    """Test that .compile() can be called early and returns cached kernel."""
    print("Test 1: Early compilation pattern")
    kernel = SoftCapMLXFastKernel()

    # Verify kernel is None before compilation
    assert kernel.kernel is None
    print("  ✓ Kernel starts as None")

    # Compile explicitly
    compiled = kernel.compile()

    # Verify kernel is now set
    assert kernel.kernel is not None
    assert kernel.kernel is compiled
    print("  ✓ Kernel compiled successfully")

    # Verify calling compile again returns same object
    compiled2 = kernel.compile()
    assert compiled2 is compiled
    print("  ✓ Kernel caching works (same object returned)")

    # Test with known values: soft_cap(x, cap) = cap * tanh(x/cap)
    # For x=0: tanh(0) = 0, so output = 0
    # For x=cap: tanh(1) ≈ 0.7616, so output ≈ cap * 0.7616
    # For x=10*cap: tanh(10) ≈ 1.0, so output ≈ cap
    # For x=-10*cap: tanh(-10) ≈ -1.0, so output ≈ -cap
    x = mx.array([0.0, 5.0, 50.0, -50.0], dtype=mx.float32)
    cap_value = 5.0
    y = kernel(x, cap_value)

    y_list = y.tolist()
    assert abs(y_list[0] - 0.0) < 1e-5
    assert abs(y_list[1] - 3.808) < 0.01
    assert abs(y_list[2] - 5.0) < 0.01
    assert abs(y_list[3] - (-5.0)) < 0.01
    print(f"  ✓ Numerical correctness: {y_list}")

def test_lazy_compilation():
    """Test that calling without compile() still works (lazy compilation)."""
    print("\nTest 2: Lazy compilation pattern")
    kernel = SoftCapMLXFastKernel()

    # Verify kernel is None before first use
    assert kernel.kernel is None
    print("  ✓ Kernel starts as None")

    # Use kernel without explicit compile()
    # Test edge case: cap=2.0
    # x=0: output=0
    # x=2: output=2*tanh(1)≈1.523
    # x=10: output≈2 (saturates at cap)
    x = mx.array([0.0, 2.0, 10.0], dtype=mx.float32)
    y = kernel(x, 2.0)

    # Verify kernel was compiled on first use
    assert kernel.kernel is not None
    print("  ✓ Kernel compiled on first use (lazy)")

    # Verify against hand-calculated values
    y_list = y.tolist()
    assert abs(y_list[0] - 0.0) < 1e-5
    assert abs(y_list[1] - 1.523) < 0.01
    assert abs(y_list[2] - 2.0) < 0.01
    print(f"  ✓ Numerical correctness: {y_list}")

if __name__ == "__main__":
    print("=" * 60)
    print("Soft Cap Kernel Compilation Tests")
    print("=" * 60)

    test_early_compilation()
    test_lazy_compilation()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

