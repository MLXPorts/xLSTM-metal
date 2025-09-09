
import os
import mlx.core as mx
from mlx_fast_kernels.qr_kernels import project_coeffs, update_vector


def test_qr_project_and_update():
    """Tests the QR projection and update kernels."""
    mx.random.seed(1)
    m, k = 128, 16
    Q = mx.random.normal(shape=(m, k), dtype=mx.float32)
    # Orthonormalize quickly via QR from numpy-like path (approx)
    # For test parity, we just compare operators, not exact orthonormality.
    v = mx.random.normal(shape=(m,), dtype=mx.float32)
    c_ref = mx.matmul(mx.transpose(Q), v)
    c = project_coeffs(Q, v)
    mx.eval(c_ref, c)
    assert float(mx.max(mx.abs(c_ref - c))) <= 1e-4
    v_ref = v - mx.matmul(Q, c)
    v_upd = update_vector(Q, c, v)
    mx.eval(v_ref, v_upd)
    assert float(mx.max(mx.abs(v_ref - v_upd))) <= 1e-4

