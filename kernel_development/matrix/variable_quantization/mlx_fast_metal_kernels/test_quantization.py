"""Test variable quantization kernel."""

import mlx.core as mx
import numpy as np
from variable_quantization import VariableQuantizationMLXKernel, quantize


def test_kernel_compilation():
    """Test that .compile() works correctly."""
    print("\nTest 1: Kernel Compilation")
    kernel = VariableQuantizationMLXKernel()

    assert kernel._kernel is None
    print("  ✓ Kernel starts as None")

    compiled = kernel.compile()
    assert kernel._kernel is not None
    assert kernel._kernel is compiled
    print("  ✓ Kernel compiled successfully")

    compiled2 = kernel.compile()
    assert compiled2 is compiled
    print("  ✓ Kernel caching works")


def test_quantization_4bit():
    """Test 4-bit quantization with known values."""
    print("\nTest 2: 4-bit Quantization")
    # For 4-bit: scale = 2^(4-1) - 1 = 7
    # Quantization levels: -1.0, -0.857, -0.714, -0.571, -0.429, -0.286, -0.143, 0.0,
    #                       0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0

    kernel = VariableQuantizationMLXKernel()

    # Test exact quantization levels
    x = mx.array([0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0], dtype=mx.float32)
    y = kernel(x, bits=4)

    y_list = y.tolist()
    expected = [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0]

    print(f"  Input:    {[f'{v:.3f}' for v in x.tolist()]}")
    print(f"  Output:   {[f'{v:.3f}' for v in y_list]}")
    print(f"  Expected: {[f'{v:.3f}' for v in expected]}")

    for i, (actual, exp) in enumerate(zip(y_list, expected)):
        assert abs(actual - exp) < 0.01, f"Index {i}: expected {exp}, got {actual}"
    print("  ✓ All values within tolerance")


def test_quantization_clipping():
    """Test that values outside [-1, 1] are clipped."""
    print("\nTest 3: Clipping Behavior")
    kernel = VariableQuantizationMLXKernel()

    # Values outside range should be clipped
    x = mx.array([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=mx.float32)
    y = kernel(x, bits=4)

    y_list = y.tolist()
    expected = [-1.0, -1.0, 0.0, 1.0, 1.0]

    print(f"  Input:    {x.tolist()}")
    print(f"  Output:   {[f'{v:.3f}' for v in y_list]}")
    print(f"  Expected: {expected}")

    # Should clip to [-1, 1]
    assert abs(y_list[0] - (-1.0)) < 0.01
    assert abs(y_list[1] - (-1.0)) < 0.01
    assert abs(y_list[2] - 0.0) < 0.01
    assert abs(y_list[3] - 1.0) < 0.01
    assert abs(y_list[4] - 1.0) < 0.01
    print("  ✓ Clipping works correctly")


def test_quantization_8bit():
    """Test 8-bit quantization (higher precision)."""
    print("\nTest 4: 8-bit Quantization")
    kernel = VariableQuantizationMLXKernel()

    # For 8-bit: scale = 2^(8-1) - 1 = 127
    # Quantization step size is 1/127 ≈ 0.0079
    x = mx.array([0.5, 0.25, 0.125], dtype=mx.float32)
    y = kernel(x, bits=8)

    y_list = y.tolist()
    print(f"  Input:  {x.tolist()}")
    print(f"  Output: {[f'{v:.4f}' for v in y_list]}")

    # 8-bit should be close to input but not exact
    # Tolerance should account for quantization step
    assert abs(y_list[0] - 0.5) < 0.01
    assert abs(y_list[1] - 0.25) < 0.01
    assert abs(y_list[2] - 0.125) < 0.01
    print("  ✓ High precision maintained")


def test_quantization_2bit():
    """Test 2-bit quantization (very coarse)."""
    print("\nTest 5: 2-bit Quantization (Coarse)")
    kernel = VariableQuantizationMLXKernel()

    # For 2-bit: scale = 2^(2-1) - 1 = 1
    # Only 3 levels: -1.0, 0.0, 1.0
    x = mx.array([-0.6, -0.2, 0.0, 0.3, 0.7], dtype=mx.float32)
    y = kernel(x, bits=2)

    y_list = y.tolist()
    expected_approx = [-1.0, 0.0, 0.0, 0.0, 1.0]

    print(f"  Input:    {x.tolist()}")
    print(f"  Output:   {[f'{v:.1f}' for v in y_list]}")
    print(f"  Expected: {expected_approx}")

    # Should snap to nearest level
    assert abs(y_list[0] - (-1.0)) < 0.01
    assert abs(y_list[1] - 0.0) < 0.01
    assert abs(y_list[2] - 0.0) < 0.01
    assert abs(y_list[3] - 0.0) < 0.01
    assert abs(y_list[4] - 1.0) < 0.01
    print("  ✓ Coarse quantization works")


def test_functional_interface():
    """Test the convenience function."""
    print("\nTest 6: Functional Interface")
    x = mx.array([0.5, 0.0, -0.5], dtype=mx.float32)
    y = quantize(x, bits=4)

    y_list = y.tolist()
    print(f"  Input:  {x.tolist()}")
    print(f"  Output: {[f'{v:.3f}' for v in y_list]}")

    assert abs(y_list[0] - 0.571) < 0.01
    assert abs(y_list[1] - 0.0) < 0.01
    assert abs(y_list[2] - (-0.571)) < 0.01
    print("  ✓ Convenience function works")


def test_multidimensional_array():
    """Test that reshaping works correctly for multi-dimensional arrays."""
    print("\nTest 7: Multi-dimensional Arrays")
    kernel = VariableQuantizationMLXKernel()

    # 2D array
    x = mx.array([[0.5, 0.0], [-0.5, 1.0]], dtype=mx.float32)
    y = kernel(x, bits=4)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Input:\n{x}")
    print(f"  Output:\n{y}")

    assert y.shape == x.shape
    assert abs(y[0, 0].item() - 0.571) < 0.01
    assert abs(y[0, 1].item() - 0.0) < 0.01
    assert abs(y[1, 0].item() - (-0.571)) < 0.01
    assert abs(y[1, 1].item() - 1.0) < 0.01
    print("  ✓ Shape preservation works")


if __name__ == "__main__":
    print("=" * 70)
    print("Variable Quantization Kernel Tests")
    print("=" * 70)

    test_kernel_compilation()
    test_quantization_4bit()
    test_quantization_clipping()
    test_quantization_8bit()
    test_quantization_2bit()
    test_functional_interface()
    test_multidimensional_array()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

