"""Smoke tests for Metal mLSTM kernels.

Tests basic compilation, execution, and output shape correctness.
"""

import mlx.core as mx
import numpy as np
from fw_kernel_recurrent import mlstm_chunkwise_recurrent_fw_C_metal
from fw_kernel_parallel import mlstm_chunkwise_parallel_fw_Hintra_metal


def test_recurrent_kernel_smoke():
    """Smoke test for recurrent forward kernel."""
    print("\n=== Testing Recurrent Forward Kernel ===")

    # Small test dimensions
    B, NH, S, DHQK, DHHV = 1, 2, 64, 32, 32
    L = 16  # Chunk size
    NC = S // L  # Number of chunks

    print(f"Dimensions: B={B}, NH={NH}, S={S}, DHQK={DHQK}, DHHV={DHHV}")
    print(f"Chunks: NC={NC}, L={L}")

    # Create input tensors
    matK = mx.random.normal((B, NH, S, DHQK), dtype=mx.float32)
    matV = mx.random.normal((B, NH, S, DHHV), dtype=mx.float32)
    vecF = mx.random.normal((B, NH, S), dtype=mx.float32)
    vecI = mx.random.normal((B, NH, S), dtype=mx.float32)

    # Initial states (optional - testing without initial states first)
    matC_initial = None
    vecN_initial = None
    scaMinter_initial = None

    print("\nRunning kernel without initial states...")
    try:
        matC_states, vecN_states, scaMinter_states = mlstm_chunkwise_recurrent_fw_C_metal(
            matK=matK,
            matV=matV,
            vecF=vecF,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaMinter_initial,
            NC=NC,
            L=L,
            siz_b_DHQK=16,
            siz_b_DHHV=16,
            save_states_every_nth_chunk=1,
        )

        # Force evaluation
        mx.eval(matC_states, vecN_states, scaMinter_states)

        print(f"✓ Kernel executed successfully")
        print(f"  matC_states shape: {matC_states.shape} (expected: ({B}, {NH}, {(NC+1)*DHQK}, {DHHV}))")
        print(f"  vecN_states shape: {vecN_states.shape} (expected: ({B}, {NH}, {(NC+1)*DHQK}))")
        print(f"  scaMinter_states shape: {scaMinter_states.shape} (expected: ({B}, {NH}, {NC+1}))")

        # Check shapes
        assert matC_states.shape == (B, NH, (NC + 1) * DHQK, DHHV), f"Wrong matC_states shape"
        assert vecN_states.shape == (B, NH, (NC + 1) * DHQK), f"Wrong vecN_states shape"
        assert scaMinter_states.shape == (B, NH, NC + 1), f"Wrong scaMinter_states shape"

        # Check for NaN/Inf
        assert not mx.any(mx.isnan(matC_states)).item(), "matC_states contains NaN"
        assert not mx.any(mx.isinf(matC_states)).item(), "matC_states contains Inf"
        assert not mx.any(mx.isnan(vecN_states)).item(), "vecN_states contains NaN"
        assert not mx.any(mx.isinf(vecN_states)).item(), "vecN_states contains Inf"
        assert not mx.any(mx.isnan(scaMinter_states)).item(), "scaMinter_states contains NaN"
        assert not mx.any(mx.isinf(scaMinter_states)).item(), "scaMinter_states contains Inf"

        print(f"✓ Output shapes correct")
        print(f"✓ No NaN/Inf values")

        # Check output statistics
        print(f"\nOutput statistics:")
        print(f"  matC_states: mean={float(mx.mean(matC_states)):.6f}, std={float(mx.std(matC_states)):.6f}")
        print(f"  vecN_states: mean={float(mx.mean(vecN_states)):.6f}, std={float(mx.std(vecN_states)):.6f}")
        print(f"  scaMinter_states: mean={float(mx.mean(scaMinter_states)):.6f}, std={float(mx.std(scaMinter_states)):.6f}")

    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with initial states
    print("\nRunning kernel with initial states...")
    try:
        matC_initial = mx.random.normal((B, NH, DHQK, DHHV), dtype=mx.float32) * 0.01
        vecN_initial = mx.random.normal((B, NH, DHQK), dtype=mx.float32) * 0.01
        scaMinter_initial = mx.random.normal((B, NH), dtype=mx.float32) * 0.01

        matC_states, vecN_states, scaMinter_states = mlstm_chunkwise_recurrent_fw_C_metal(
            matK=matK,
            matV=matV,
            vecF=vecF,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaMinter_initial,
            NC=NC,
            L=L,
            siz_b_DHQK=16,
            siz_b_DHHV=16,
            save_states_every_nth_chunk=1,
        )

        mx.eval(matC_states, vecN_states, scaMinter_states)

        print(f"✓ Kernel with initial states executed successfully")
        assert matC_states.shape == (B, NH, (NC + 1) * DHQK, DHHV)
        print(f"✓ Shapes correct with initial states")

    except Exception as e:
        print(f"✗ Kernel with initial states failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓✓✓ Recurrent kernel smoke test PASSED ✓✓✓")
    return True


def test_parallel_kernel_smoke():
    """Smoke test for parallel forward kernel."""
    print("\n=== Testing Parallel Forward Kernel ===")

    # Small test dimensions
    B, NH, S, DHQK, DHHV = 1, 2, 64, 32, 32
    L = 16  # Chunk size
    NC = S // L  # Number of chunks
    qk_scale = DHQK ** (-0.5)

    print(f"Dimensions: B={B}, NH={NH}, S={S}, DHQK={DHQK}, DHHV={DHHV}")
    print(f"Chunks: NC={NC}, L={L}, qk_scale={qk_scale}")

    # Create input tensors
    matQ = mx.random.normal((B, NH, S, DHQK), dtype=mx.float32) * 0.1
    matK = mx.random.normal((B, NH, S, DHQK), dtype=mx.float32) * 0.1
    matV = mx.random.normal((B, NH, S, DHHV), dtype=mx.float32) * 0.1

    # Create states from recurrent kernel (simulated)
    matC_states = mx.random.normal((B, NH, (NC + 1) * DHQK, DHHV), dtype=mx.float32) * 0.01
    vecN_states = mx.random.normal((B, NH, (NC + 1) * DHQK), dtype=mx.float32) * 0.01
    scaMinter_states = mx.random.normal((B, NH, NC + 1), dtype=mx.float32) * 0.01

    # Gates (reshaped to (B, NH, NC, L))
    vecI = mx.random.normal((B, NH, NC, L), dtype=mx.float32)
    vecF = mx.random.normal((B, NH, NC, L), dtype=mx.float32)

    # Compute vecB = cumsum(logsigmoid(vecF))
    vecF_logsig = -mx.log(1.0 + mx.exp(-vecF))
    vecB = mx.cumsum(vecF_logsig, axis=-1)

    print("\nRunning parallel kernel...")
    try:
        matHout, vecNout, vecMout = mlstm_chunkwise_parallel_fw_Hintra_metal(
            matQ=matQ,
            matK=matK,
            matV=matV,
            matC_states=matC_states,
            vecN_states=vecN_states,
            scaMinter_states=scaMinter_states,
            vecI=vecI,
            vecB=vecB,
            NC=NC,
            L=L,
            qk_scale=qk_scale,
            siz_b_LQ=16,
            siz_b_LKV=16,
            siz_b_DHQK=16,
            siz_b_DHHV=16,
            eps=1e-6,
            minimum_max_val=-10.0,
        )

        # Force evaluation
        mx.eval(matHout, vecNout, vecMout)

        print(f"✓ Kernel executed successfully")
        print(f"  matHout shape: {matHout.shape} (expected: ({B}, {NH}, {S}, {DHHV}))")
        print(f"  vecNout shape: {vecNout.shape} (expected: ({B}, {NH}, {S}))")
        print(f"  vecMout shape: {vecMout.shape} (expected: ({B}, {NH}, {S}))")

        # Check shapes
        assert matHout.shape == (B, NH, S, DHHV), f"Wrong matHout shape"
        assert vecNout.shape == (B, NH, S), f"Wrong vecNout shape"
        assert vecMout.shape == (B, NH, S), f"Wrong vecMout shape"

        # Check for NaN/Inf
        assert not mx.any(mx.isnan(matHout)).item(), "matHout contains NaN"
        assert not mx.any(mx.isinf(matHout)).item(), "matHout contains Inf"
        assert not mx.any(mx.isnan(vecNout)).item(), "vecNout contains NaN"
        assert not mx.any(mx.isinf(vecNout)).item(), "vecNout contains Inf"
        assert not mx.any(mx.isnan(vecMout)).item(), "vecMout contains NaN"
        assert not mx.any(mx.isinf(vecMout)).item(), "vecMout contains Inf"

        print(f"✓ Output shapes correct")
        print(f"✓ No NaN/Inf values")

        # Check output statistics
        print(f"\nOutput statistics:")
        print(f"  matHout: mean={float(mx.mean(matHout)):.6f}, std={float(mx.std(matHout)):.6f}")
        print(f"  vecNout: mean={float(mx.mean(vecNout)):.6f}, std={float(mx.std(vecNout)):.6f}")
        print(f"  vecMout: mean={float(mx.mean(vecMout)):.6f}, std={float(mx.std(vecMout)):.6f}")

    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓✓✓ Parallel kernel smoke test PASSED ✓✓✓")
    return True


def test_different_tile_sizes():
    """Test kernels with different tile sizes."""
    print("\n=== Testing Different Tile Sizes ===")

    B, NH, S, DHQK, DHHV = 1, 1, 32, 16, 16
    L = 16
    NC = S // L

    tile_configs = [
        (8, 8),
        (16, 16),
    ]

    for siz_b_DHQK, siz_b_DHHV in tile_configs:
        print(f"\nTesting with tile size: ({siz_b_DHQK}, {siz_b_DHHV})")

        try:
            matK = mx.random.normal((B, NH, S, DHQK), dtype=mx.float32) * 0.1
            matV = mx.random.normal((B, NH, S, DHHV), dtype=mx.float32) * 0.1
            vecF = mx.random.normal((B, NH, S), dtype=mx.float32)
            vecI = mx.random.normal((B, NH, S), dtype=mx.float32)

            matC_states, vecN_states, scaMinter_states = mlstm_chunkwise_recurrent_fw_C_metal(
                matK=matK,
                matV=matV,
                vecF=vecF,
                vecI=vecI,
                matC_initial=None,
                vecN_initial=None,
                scaMinter_initial=None,
                NC=NC,
                L=L,
                siz_b_DHQK=siz_b_DHQK,
                siz_b_DHHV=siz_b_DHHV,
            )

            mx.eval(matC_states)
            print(f"  ✓ Recurrent kernel works with tile ({siz_b_DHQK}, {siz_b_DHHV})")

        except Exception as e:
            print(f"  ✗ Failed with tile ({siz_b_DHQK}, {siz_b_DHHV}): {e}")
            return False

    print("\n✓ Different tile sizes test PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Metal mLSTM Kernel Smoke Tests")
    print("=" * 60)

    all_passed = True

    # Test recurrent kernel
    if not test_recurrent_kernel_smoke():
        all_passed = False

    # Test parallel kernel
    if not test_parallel_kernel_smoke():
        all_passed = False

    # Test different tile sizes
    if not test_different_tile_sizes():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SMOKE TESTS PASSED ✓✓✓")
    else:
        print("SOME TESTS FAILED ✗✗✗")
    print("=" * 60)
