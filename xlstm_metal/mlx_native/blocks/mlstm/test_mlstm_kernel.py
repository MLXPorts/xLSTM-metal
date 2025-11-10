"""
Unit tests for mLSTM kernel (exponential gating recurrence)

Tests the core mlstm_recurrent_step and mlstm_sequential functions.
"""

import mlx.core as mx

from kernel import mlstm_recurrent_step, mlstm_sequential


def test_recurrent_step_shapes():
    """Test mLSTM recurrent step produces correct output shapes"""
    print("\n" + "=" * 80)
    print("Test 1: mLSTM Recurrent Step Shapes")
    print("=" * 80)

    B, NH, DH = 2, 4, 256

    # Inputs
    q = mx.random.normal(shape=(B, NH, DH))
    k = mx.random.normal(shape=(B, NH, DH))
    v = mx.random.normal(shape=(B, NH, DH))
    i_preact = mx.random.normal(shape=(B, NH))
    f_preact = mx.random.normal(shape=(B, NH))

    # States
    c_state = mx.zeros((B, NH, DH, DH))
    n_state = mx.zeros((B, NH, DH))
    m_state = mx.zeros((B, NH))

    print(f"Input shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  i_preact: {i_preact.shape}")
    print(f"  f_preact: {f_preact.shape}")
    print(f"  c_state: {c_state.shape}")
    print(f"  n_state: {n_state.shape}")
    print(f"  m_state: {m_state.shape}")

    # Run step
    h, c_new, n_new, m_new = mlstm_recurrent_step(
        q, k, v, i_preact, f_preact,
        c_state, n_state, m_state
    )

    print(f"\nOutput shapes:")
    print(f"  h: {h.shape}")
    print(f"  c_new: {c_new.shape}")
    print(f"  n_new: {n_new.shape}")
    print(f"  m_new: {m_new.shape}")

    # Validate shapes
    assert h.shape == (B, NH, DH), f"h shape mismatch: {h.shape}"
    assert c_new.shape == (B, NH, DH, DH), f"c_new shape mismatch: {c_new.shape}"
    assert n_new.shape == (B, NH, DH), f"n_new shape mismatch: {n_new.shape}"
    assert m_new.shape == (B, NH), f"m_new shape mismatch: {m_new.shape}"

    print("✓ All output shapes correct")
    print()


def test_exponential_gating_stability():
    """Test exponential gating numerical stability"""
    print("=" * 80)
    print("Test 2: Exponential Gating Numerical Stability")
    print("=" * 80)

    B, NH, DH = 1, 2, 64

    # Test with large pre-activations (should not overflow)
    q = mx.random.normal(shape=(B, NH, DH))
    k = mx.random.normal(shape=(B, NH, DH))
    v = mx.random.normal(shape=(B, NH, DH))
    i_preact = mx.array([[10.0, 15.0]])  # Large values
    f_preact = mx.array([[20.0, 25.0]])  # Large values

    c_state = mx.zeros((B, NH, DH, DH))
    n_state = mx.zeros((B, NH, DH))
    m_state = mx.zeros((B, NH))

    print(f"Testing with large pre-activations:")
    print(f"  i_preact: {i_preact.tolist()}")
    print(f"  f_preact: {f_preact.tolist()}")

    h, c_new, n_new, m_new = mlstm_recurrent_step(
        q, k, v, i_preact, f_preact,
        c_state, n_state, m_state
    )

    # Check for NaN or Inf
    h_has_nan = mx.any(mx.isnan(h))
    h_has_inf = mx.any(mx.isinf(h))

    print(f"\nOutput checks:")
    print(f"  h contains NaN: {bool(h_has_nan)}")
    print(f"  h contains Inf: {bool(h_has_inf)}")
    print(f"  h max: {float(mx.max(mx.abs(h))):.6f}")
    print(f"  m_new: {m_new.tolist()}")

    assert not bool(h_has_nan), "Output contains NaN"
    assert not bool(h_has_inf), "Output contains Inf"

    # m_new should be the running max
    expected_m = mx.maximum(f_preact + m_state, i_preact)
    m_diff = mx.max(mx.abs(m_new - expected_m))
    print(f"  m_new correctness (max diff): {float(m_diff):.10f}")
    assert float(m_diff) < 1e-5, "Running max computation incorrect"

    print("✓ Exponential gating numerically stable")
    print()


def test_sequential_processing():
    """Test sequential processing over multiple timesteps"""
    print("=" * 80)
    print("Test 3: Sequential Processing")
    print("=" * 80)

    B, NH, S, DH = 2, 4, 8, 128

    # Inputs
    q = mx.random.normal(shape=(B, NH, S, DH))
    k = mx.random.normal(shape=(B, NH, S, DH))
    v = mx.random.normal(shape=(B, NH, S, DH))
    i_preact = mx.random.normal(shape=(B, NH, S))
    f_preact = mx.random.normal(shape=(B, NH, S))

    print(f"Input shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  i_preact: {i_preact.shape}")
    print(f"  f_preact: {f_preact.shape}")

    # Run sequential
    h, (c_final, n_final, m_final) = mlstm_sequential(
        q, k, v, i_preact, f_preact
    )

    print(f"\nOutput shapes:")
    print(f"  h: {h.shape}")
    print(f"  c_final: {c_final.shape}")
    print(f"  n_final: {n_final.shape}")
    print(f"  m_final: {m_final.shape}")

    # Validate shapes
    assert h.shape == (B, NH, S, DH), f"h shape mismatch: {h.shape}"
    assert c_final.shape == (B, NH, DH, DH), f"c_final shape mismatch: {c_final.shape}"
    assert n_final.shape == (B, NH, DH), f"n_final shape mismatch: {n_final.shape}"
    assert m_final.shape == (B, NH), f"m_final shape mismatch: {m_final.shape}"

    # Check for NaN/Inf
    h_has_nan = mx.any(mx.isnan(h))
    h_has_inf = mx.any(mx.isinf(h))

    print(f"\nNumerical checks:")
    print(f"  h contains NaN: {bool(h_has_nan)}")
    print(f"  h contains Inf: {bool(h_has_inf)}")
    print(f"  h mean: {float(mx.mean(h)):.6f}")
    print(f"  h std: {float(mx.std(h)):.6f}")

    assert not bool(h_has_nan), "Output contains NaN"
    assert not bool(h_has_inf), "Output contains Inf"

    print("✓ Sequential processing works correctly")
    print()


def test_state_continuity():
    """Test that states can be carried across batches"""
    print("=" * 80)
    print("Test 4: State Continuity")
    print("=" * 80)

    B, NH, S1, S2, DH = 1, 2, 4, 4, 64

    # First sequence
    q1 = mx.random.normal(shape=(B, NH, S1, DH))
    k1 = mx.random.normal(shape=(B, NH, S1, DH))
    v1 = mx.random.normal(shape=(B, NH, S1, DH))
    i1 = mx.random.normal(shape=(B, NH, S1))
    f1 = mx.random.normal(shape=(B, NH, S1))

    h1, (c1, n1, m1) = mlstm_sequential(q1, k1, v1, i1, f1)

    print(f"First sequence (S={S1}):")
    print(f"  Final states: c={c1.shape}, n={n1.shape}, m={m1.shape}")

    # Second sequence (continuing from first)
    q2 = mx.random.normal(shape=(B, NH, S2, DH))
    k2 = mx.random.normal(shape=(B, NH, S2, DH))
    v2 = mx.random.normal(shape=(B, NH, S2, DH))
    i2 = mx.random.normal(shape=(B, NH, S2))
    f2 = mx.random.normal(shape=(B, NH, S2))

    h2, (c2, n2, m2) = mlstm_sequential(
        q2, k2, v2, i2, f2,
        c_initial=c1, n_initial=n1, m_initial=m1
    )

    print(f"Second sequence (S={S2}, continuing from first):")
    print(f"  Output shape: {h2.shape}")
    print(f"  Final states: c={c2.shape}, n={n2.shape}, m={m2.shape}")

    # Concatenate sequences and run in one pass
    q_full = mx.concatenate([q1, q2], axis=2)  # [B, NH, S1+S2, DH]
    k_full = mx.concatenate([k1, k2], axis=2)
    v_full = mx.concatenate([v1, v2], axis=2)
    i_full = mx.concatenate([i1, i2], axis=2)
    f_full = mx.concatenate([f1, f2], axis=2)

    h_full, (c_full, n_full, m_full) = mlstm_sequential(
        q_full, k_full, v_full, i_full, f_full
    )

    print(f"Full sequence (S={S1+S2}):")
    print(f"  Output shape: {h_full.shape}")

    # Compare: second half of full run vs continued run
    h2_from_full = h_full[:, :, S1:, :]
    diff = mx.max(mx.abs(h2 - h2_from_full))

    print(f"\nState continuity check:")
    print(f"  Max diff between continued vs full: {float(diff):.10f}")

    # Should be identical (or very close)
    assert float(diff) < 1e-4, f"State continuity broken: diff={float(diff)}"

    print("✓ State continuity preserved")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("mLSTM Kernel Unit Tests")
    print("=" * 80)

    try:
        test_recurrent_step_shapes()
        test_exponential_gating_stability()
        test_sequential_processing()
        test_state_continuity()

        print("=" * 80)
        print("All mLSTM Kernel Tests Passed! ✓")
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
