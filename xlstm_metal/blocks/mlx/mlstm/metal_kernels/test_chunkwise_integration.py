"""Test chunkwise mLSTM integration with Metal kernels.

Verifies that the chunkwise implementation produces reasonable outputs.
"""

import mlx.core as mx
import sys
sys.path.insert(0, '/Volumes/emberstuff/xLSTM/mad/blocks')

from mlstm_mlx.kernel import mlstm_chunkwise, mlstm_sequential


def test_chunkwise_basic():
    """Test basic chunkwise execution."""
    print("\n=== Testing Chunkwise Basic Execution ===")

    # Small test dimensions
    B, NH, S, QK_DH, V_DH = 1, 2, 128, 32, 32
    chunk_size = 64

    print(f"Dimensions: B={B}, NH={NH}, S={S}, QK_DH={QK_DH}, V_DH={V_DH}")
    print(f"Chunk size: {chunk_size}")

    # Create random inputs
    q = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
    k = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
    v = mx.random.normal((B, NH, S, V_DH), dtype=mx.float32) * 0.1
    i_preact = mx.random.normal((B, NH, S), dtype=mx.float32)
    f_preact = mx.random.normal((B, NH, S), dtype=mx.float32)

    print("\nRunning chunkwise kernel...")
    try:
        h, (c_final, n_final, m_final) = mlstm_chunkwise(
            q=q,
            k=k,
            v=v,
            i_preact=i_preact,
            f_preact=f_preact,
            chunk_size=chunk_size,
            c_initial=None,
            n_initial=None,
            m_initial=None,
            eps=1e-6,
            return_last_states=True,
        )

        # Force evaluation
        mx.eval(h, c_final, n_final, m_final)

        print(f"✓ Chunkwise executed successfully")
        print(f"  h shape: {h.shape} (expected: ({B}, {NH}, {S}, {V_DH}))")
        print(f"  c_final shape: {c_final.shape} (expected: ({B}, {NH}, {QK_DH}, {V_DH}))")
        print(f"  n_final shape: {n_final.shape} (expected: ({B}, {NH}, {QK_DH}))")
        print(f"  m_final shape: {m_final.shape} (expected: ({B}, {NH}))")

        # Check shapes
        assert h.shape == (B, NH, S, V_DH), f"Wrong h shape"
        assert c_final.shape == (B, NH, QK_DH, V_DH), f"Wrong c_final shape"
        assert n_final.shape == (B, NH, QK_DH), f"Wrong n_final shape"
        assert m_final.shape == (B, NH), f"Wrong m_final shape"

        # Check for NaN/Inf
        assert not mx.any(mx.isnan(h)).item(), "h contains NaN"
        assert not mx.any(mx.isinf(h)).item(), "h contains Inf"
        assert not mx.any(mx.isnan(c_final)).item(), "c_final contains NaN"
        assert not mx.any(mx.isinf(c_final)).item(), "c_final contains Inf"

        print(f"✓ Output shapes correct")
        print(f"✓ No NaN/Inf values")

        # Check output statistics
        print(f"\nOutput statistics:")
        print(f"  h: mean={float(mx.mean(h)):.6f}, std={float(mx.std(h)):.6f}")
        print(f"  c_final: mean={float(mx.mean(c_final)):.6f}, std={float(mx.std(c_final)):.6f}")
        print(f"  n_final: mean={float(mx.mean(n_final)):.6f}, std={float(mx.std(n_final)):.6f}")
        print(f"  m_final: mean={float(mx.mean(m_final)):.6f}, std={float(mx.std(m_final)):.6f}")

        return True

    except Exception as e:
        print(f"✗ Chunkwise failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunkwise_with_initial_states():
    """Test chunkwise with initial states."""
    print("\n=== Testing Chunkwise With Initial States ===")

    B, NH, S, QK_DH, V_DH = 1, 2, 64, 32, 32
    chunk_size = 32

    print(f"Dimensions: B={B}, NH={NH}, S={S}, QK_DH={QK_DH}, V_DH={V_DH}")

    # Create inputs
    q = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
    k = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
    v = mx.random.normal((B, NH, S, V_DH), dtype=mx.float32) * 0.1
    i_preact = mx.random.normal((B, NH, S), dtype=mx.float32)
    f_preact = mx.random.normal((B, NH, S), dtype=mx.float32)

    # Create initial states
    c_initial = mx.random.normal((B, NH, QK_DH, V_DH), dtype=mx.float32) * 0.01
    n_initial = mx.random.normal((B, NH, QK_DH), dtype=mx.float32) * 0.01
    m_initial = mx.random.normal((B, NH), dtype=mx.float32) * 0.01

    print("\nRunning chunkwise with initial states...")
    try:
        h, (c_final, n_final, m_final) = mlstm_chunkwise(
            q=q,
            k=k,
            v=v,
            i_preact=i_preact,
            f_preact=f_preact,
            chunk_size=chunk_size,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            eps=1e-6,
            return_last_states=True,
        )

        mx.eval(h, c_final, n_final, m_final)

        print(f"✓ Chunkwise with initial states executed successfully")
        assert h.shape == (B, NH, S, V_DH)
        print(f"✓ Shapes correct with initial states")

        return True

    except Exception as e:
        print(f"✗ Chunkwise with initial states failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunkwise_different_chunk_sizes():
    """Test chunkwise with different chunk sizes."""
    print("\n=== Testing Different Chunk Sizes ===")

    B, NH, S, QK_DH, V_DH = 1, 1, 128, 16, 16

    chunk_sizes = [16, 32, 64]

    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk_size={chunk_size}...")

        try:
            q = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
            k = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
            v = mx.random.normal((B, NH, S, V_DH), dtype=mx.float32) * 0.1
            i_preact = mx.random.normal((B, NH, S), dtype=mx.float32)
            f_preact = mx.random.normal((B, NH, S), dtype=mx.float32)

            h, _ = mlstm_chunkwise(
                q=q, k=k, v=v,
                i_preact=i_preact,
                f_preact=f_preact,
                chunk_size=chunk_size,
                return_last_states=False,
            )

            mx.eval(h)
            print(f"  ✓ chunk_size={chunk_size} works")

        except Exception as e:
            print(f"  ✗ chunk_size={chunk_size} failed: {e}")
            return False

    print("\n✓ All chunk sizes work")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Chunkwise mLSTM Integration Tests")
    print("=" * 60)

    all_passed = True

    # Test basic execution
    if not test_chunkwise_basic():
        all_passed = False

    # Test with initial states
    if not test_chunkwise_with_initial_states():
        all_passed = False

    # Test different chunk sizes
    if not test_chunkwise_different_chunk_sizes():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED ✓✓✓")
    else:
        print("SOME TESTS FAILED ✗✗✗")
    print("=" * 60)
