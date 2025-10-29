"""Test PyTorch Metal soft_cap implementation."""

import torch
from softcap import metal_soft_cap


def test_metal_soft_cap():
    """Test the Metal soft_cap kernel."""
    print("=" * 70)
    print("PyTorch Metal Soft Cap Test")
    print("=" * 70)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("✗ MPS not available - skipping test")
        return

    print("\n✓ MPS is available")

    # Create test tensor on MPS
    x = torch.tensor([0.0, 5.0, 50.0, -50.0, 0.5, -0.5], device="mps", dtype=torch.float32)
    cap_value = 5.0

    print(f"\nTest 1: Basic soft cap with cap={cap_value}")
    print(f"  Input:  {x.cpu().tolist()}")

    try:
        y = metal_soft_cap(x, cap_value)
        y_cpu = y.cpu().tolist()
        print(f"  Output: {y_cpu}")

        # Expected values
        # 0.0 -> 0.0
        # 5.0 -> ~3.808 (5 * tanh(1))
        # 50.0 -> ~5.0 (saturates)
        # -50.0 -> ~-5.0 (saturates)
        # 0.5 -> ~0.5 (linear region)
        # -0.5 -> ~-0.5 (linear region)

        expected = [0.0, 3.808, 5.0, -5.0, 0.5, -0.5]
        print(f"  Expected: {expected}")

        # Validate
        for i, (actual, exp) in enumerate(zip(y_cpu, expected)):
            if abs(actual - exp) > 0.05:
                print(f"  ✗ Index {i}: expected {exp:.3f}, got {actual:.3f}")
                return

        print("  ✓ All values correct")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Different shapes
    print(f"\nTest 2: 2D tensor")
    x2d = torch.tensor([[0.5, 1.0], [2.0, 3.0]], device="mps", dtype=torch.float32)
    print(f"  Input shape: {x2d.shape}")

    try:
        y2d = metal_soft_cap(x2d, 5.0)
        print(f"  Output shape: {y2d.shape}")
        print(f"  Output:\n{y2d.cpu()}")
        print("  ✓ Multi-dimensional works")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    # Test 3: Large tensor
    print(f"\nTest 3: Large tensor (10000 elements)")
    x_large = torch.randn(10000, device="mps", dtype=torch.float32)

    try:
        y_large = metal_soft_cap(x_large, 2.0)

        # Check all values are within [-cap, cap]
        y_cpu_large = y_large.cpu()
        if (y_cpu_large.abs() <= 2.01).all():
            print(f"  ✓ All values bounded by cap")
        else:
            print(f"  ✗ Some values exceed cap")
            return

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_metal_soft_cap()

