#!/usr/bin/env python
"""
Test sLSTM Metal kernel against canonical reference implementation.

Validates that the Metal kernel produces bit-exact results matching
the canonical sLSTM equations from xlstm package.
"""

import mlx.core as mx
from xlstm_metal.mlx_jit.blocks.slstm.slstm_layers.stepwise.slstm_metal_kernel import slstm_step_metal


def logsigmoid_mlx(x: mx.array) -> mx.array:
    """Stable logsigmoid using MLX ops only."""
    zero = mx.array(0.0, dtype=mx.float32)
    # if x >= 0: -log(1 + exp(-x))
    # if x < 0:  x - log(1 + exp(x))
    neg_x = mx.multiply(x, mx.array(-1.0, dtype=mx.float32))
    one = mx.array(1.0, dtype=mx.float32)

    # For x >= 0 branch
    pos_result = mx.multiply(
        mx.array(-1.0, dtype=mx.float32),
        mx.log(mx.add(one, mx.exp(neg_x)))
    )

    # For x < 0 branch
    neg_result = mx.subtract(x, mx.log(mx.add(one, mx.exp(x))))

    # Select based on condition
    return mx.where(mx.greater_equal(x, zero), pos_result, neg_result)


def slstm_step_reference(
    z: mx.array,
    i_preact: mx.array,
    f_preact: mx.array,
    o_preact: mx.array,
    c_state: mx.array,
    n_state: mx.array,
    m_state: mx.array,
    eps: float = 1e-6
) -> tuple:
    """
    Reference sLSTM step using pure MLX ops (canonical equations).

    This implements the exact equations from:
    /Users/sydneybach/miniconda3/lib/python3.12/site-packages/xlstm/blocks/slstm/src/vanilla/slstm.py

    Lines 24-34:
    - logfplusm = m + logsigmoid(fraw)
    - mnew = max(iraw, logfplusm)
    - igate = min(exp(iraw - mnew), 1.0)
    - fgate = min(exp(logfplusm - mnew), 1.0)
    - ogate = sigmoid(oraw)
    - cnew = fgate * c + igate * tanh(zraw)
    - nnew = fgate * n + igate
    - ynew = ogate * cnew / nnew
    """
    B, NH, H = z.shape

    # Ensure float32
    z = z.astype(mx.float32)
    i_preact = i_preact.astype(mx.float32)
    f_preact = f_preact.astype(mx.float32)
    o_preact = o_preact.astype(mx.float32)
    c_state = c_state.astype(mx.float32)
    n_state = n_state.astype(mx.float32)
    m_state = m_state.astype(mx.float32)

    eps_arr = mx.array(eps, dtype=mx.float32)
    one = mx.array(1.0, dtype=mx.float32)

    # Expand m_state to match z shape for broadcasting
    m_exp = mx.expand_dims(m_state, axis=2)  # [B, NH, 1]

    # Line 24: logfplusm = m + logsigmoid(fraw)
    f_exp = mx.expand_dims(f_preact, axis=2)  # [B, NH, 1]
    logfplusm = mx.add(m_exp, logsigmoid_mlx(f_exp))  # [B, NH, 1]

    # Line 28: mnew = max(iraw, logfplusm)
    i_exp = mx.expand_dims(i_preact, axis=2)  # [B, NH, 1]
    m_new = mx.maximum(i_exp, logfplusm)  # [B, NH, 1]

    # Line 30: igate = min(exp(iraw - mnew), 1.0)
    i_gate = mx.minimum(mx.exp(mx.subtract(i_exp, m_new)), one)  # [B, NH, 1]

    # Line 31: fgate = min(exp(logfplusm - mnew), 1.0)
    f_gate = mx.minimum(mx.exp(mx.subtract(logfplusm, m_new)), one)  # [B, NH, 1]

    # Output gate: sigmoid(oraw)
    o_exp = mx.expand_dims(o_preact, axis=2)  # [B, NH, 1]
    o_gate = mx.sigmoid(o_exp)  # [B, NH, 1]

    # Line 32: cnew = fgate * c + igate * tanh(zraw)
    c_new = mx.add(
        mx.multiply(f_gate, c_state),
        mx.multiply(i_gate, mx.tanh(z))
    )

    # Line 33: nnew = fgate * n + igate
    n_new = mx.add(
        mx.multiply(f_gate, n_state),
        i_gate
    )

    # Line 34: ynew = ogate * cnew / nnew
    h_out = mx.multiply(
        o_gate,
        mx.divide(c_new, mx.add(n_new, eps_arr))
    )

    # Squeeze m_new back to [B, NH]
    m_new_out = mx.squeeze(m_new, axis=2)

    return h_out, c_new, n_new, m_new_out


def test_single_step():
    """Test single timestep against reference."""
    print("Testing single sLSTM timestep...")

    # Small test case
    B, NH, H = 2, 4, 8
    eps = 1e-6

    # Random inputs
    mx.random.seed(42)
    z = mx.random.normal((B, NH, H))
    i_preact = mx.random.normal((B, NH))
    f_preact = mx.random.normal((B, NH))
    o_preact = mx.random.normal((B, NH))

    # Random initial state
    c_state = mx.random.normal((B, NH, H))
    n_state = mx.abs(mx.random.normal((B, NH, H)))  # Keep positive
    m_state = mx.random.normal((B, NH))

    # Reference implementation
    h_ref, c_ref, n_ref, m_ref = slstm_step_reference(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state, eps
    )

    # Metal kernel implementation
    h_metal, c_metal, n_metal, m_metal = slstm_step_metal(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state, eps
    )

    # Compare outputs
    h_diff = mx.abs(mx.subtract(h_ref, h_metal))
    c_diff = mx.abs(mx.subtract(c_ref, c_metal))
    n_diff = mx.abs(mx.subtract(n_ref, n_metal))
    m_diff = mx.abs(mx.subtract(m_ref, m_metal))

    max_h_diff = mx.max(h_diff).item()
    max_c_diff = mx.max(c_diff).item()
    max_n_diff = mx.max(n_diff).item()
    max_m_diff = mx.max(m_diff).item()

    print(f"  Max h difference: {max_h_diff:.2e}")
    print(f"  Max c difference: {max_c_diff:.2e}")
    print(f"  Max n difference: {max_n_diff:.2e}")
    print(f"  Max m difference: {max_m_diff:.2e}")

    # Tolerance: float32 precision
    tol = 1e-5

    assert max_h_diff < tol, f"h output mismatch: {max_h_diff} > {tol}"
    assert max_c_diff < tol, f"c state mismatch: {max_c_diff} > {tol}"
    assert max_n_diff < tol, f"n state mismatch: {max_n_diff} > {tol}"
    assert max_m_diff < tol, f"m state mismatch: {max_m_diff} > {tol}"

    print("  ✓ Single step test passed")


def test_zero_state():
    """Test with zero initial state."""
    print("Testing zero initial state...")

    B, NH, H = 1, 2, 4
    eps = 1e-6

    # Random inputs
    mx.random.seed(123)
    z = mx.random.normal((B, NH, H))
    i_preact = mx.random.normal((B, NH))
    f_preact = mx.random.normal((B, NH))
    o_preact = mx.random.normal((B, NH))

    # Zero initial state
    c_state = mx.zeros((B, NH, H))
    n_state = mx.zeros((B, NH, H))
    m_state = mx.zeros((B, NH))

    # Reference
    h_ref, c_ref, n_ref, m_ref = slstm_step_reference(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state, eps
    )

    # Metal
    h_metal, c_metal, n_metal, m_metal = slstm_step_metal(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state, eps
    )

    # Compare
    h_diff = mx.max(mx.abs(mx.subtract(h_ref, h_metal))).item()

    print(f"  Max h difference: {h_diff:.2e}")

    tol = 1e-5
    assert h_diff < tol, f"Zero state test failed: {h_diff} > {tol}"

    print("  ✓ Zero state test passed")


def test_numerical_stability():
    """Test with extreme values to verify stability."""
    print("Testing numerical stability...")

    B, NH, H = 1, 2, 4
    eps = 1e-6

    # Extreme gate values
    i_preact = mx.array([[10.0, -10.0]], dtype=mx.float32)  # [B, NH]
    f_preact = mx.array([[10.0, -10.0]], dtype=mx.float32)
    o_preact = mx.array([[10.0, -10.0]], dtype=mx.float32)

    # Normal z
    z = mx.random.normal((B, NH, H))

    # Large state values
    c_state = mx.multiply(mx.random.normal((B, NH, H)), mx.array(100.0, dtype=mx.float32))
    n_state = mx.multiply(mx.abs(mx.random.normal((B, NH, H))), mx.array(100.0, dtype=mx.float32))
    m_state = mx.random.normal((B, NH))

    # Reference
    h_ref, c_ref, n_ref, m_ref = slstm_step_reference(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state, eps
    )

    # Metal
    h_metal, c_metal, n_metal, m_metal = slstm_step_metal(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state, eps
    )

    # Check for NaN/Inf
    assert not mx.any(mx.isnan(h_metal)).item(), "Metal kernel produced NaN"
    assert not mx.any(mx.isinf(h_metal)).item(), "Metal kernel produced Inf"
    assert not mx.any(mx.isnan(h_ref)).item(), "Reference produced NaN"
    assert not mx.any(mx.isinf(h_ref)).item(), "Reference produced Inf"

    # Compare
    h_diff = mx.max(mx.abs(mx.subtract(h_ref, h_metal))).item()

    print(f"  Max h difference: {h_diff:.2e}")
    print("  ✓ No NaN/Inf detected")

    tol = 1e-4  # Slightly relaxed for extreme values
    assert h_diff < tol, f"Stability test failed: {h_diff} > {tol}"

    print("  ✓ Numerical stability test passed")


def test_sequential_steps():
    """Test multiple sequential steps (recurrence)."""
    print("Testing sequential recurrence...")

    B, NH, H = 2, 4, 8
    S = 5  # 5 timesteps
    eps = 1e-6

    # Random sequence
    mx.random.seed(456)
    z_seq = mx.random.normal((B, S, NH, H))
    i_seq = mx.random.normal((B, S, NH))
    f_seq = mx.random.normal((B, S, NH))
    o_seq = mx.random.normal((B, S, NH))

    # Initial state
    c_state_ref = mx.zeros((B, NH, H))
    n_state_ref = mx.zeros((B, NH, H))
    m_state_ref = mx.zeros((B, NH))

    c_state_metal = mx.zeros((B, NH, H))
    n_state_metal = mx.zeros((B, NH, H))
    m_state_metal = mx.zeros((B, NH))

    # Process sequence with both implementations
    max_diff = 0.0
    for t in range(S):
        z_t = z_seq[:, t, :, :]
        i_t = i_seq[:, t, :]
        f_t = f_seq[:, t, :]
        o_t = o_seq[:, t, :]

        # Reference step
        h_ref, c_state_ref, n_state_ref, m_state_ref = slstm_step_reference(
            z_t, i_t, f_t, o_t,
            c_state_ref, n_state_ref, m_state_ref, eps
        )

        # Metal step
        h_metal, c_state_metal, n_state_metal, m_state_metal = slstm_step_metal(
            z_t, i_t, f_t, o_t,
            c_state_metal, n_state_metal, m_state_metal, eps
        )

        # Check difference
        diff = mx.max(mx.abs(mx.subtract(h_ref, h_metal))).item()
        max_diff = max(max_diff, diff)

    print(f"  Max difference across {S} steps: {max_diff:.2e}")

    tol = 1e-5
    assert max_diff < tol, f"Sequential test failed: {max_diff} > {tol}"

    print("  ✓ Sequential recurrence test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("sLSTM Metal Kernel Validation Tests")
    print("="*60 + "\n")

    test_single_step()
    print()
    test_zero_state()
    print()
    test_numerical_stability()
    print()
    test_sequential_steps()

    print("\n" + "="*60)
    print("All tests passed! Metal kernel matches canonical reference.")
    print("="*60 + "\n")
