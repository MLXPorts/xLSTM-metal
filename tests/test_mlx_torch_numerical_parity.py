import numpy as np
import pytest
import mlx.core as mx

torch = pytest.importorskip("torch")


def test_logsigmoid_parity():
    """Logsigmoid should match PyTorch reference."""
    x_np = np.array([-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0], dtype=np.float32)

    x_torch = torch.from_numpy(x_np)
    ref = torch.nn.functional.logsigmoid(x_torch).numpy()

    x_mlx = mx.array(x_np)
    mlx_res = mx.negative(mx.log(mx.add(1.0, mx.exp(mx.negative(x_mlx)))))
    np.testing.assert_allclose(np.array(mlx_res), ref, rtol=1e-6, atol=1e-6)


def test_gate_normalization_parity():
    """Check stabilized gate math against transformers implementation."""
    i_vals = np.array([[-10.0], [0.0], [10.0]], dtype=np.float32)
    f_vals = np.array([[-10.0], [0.0], [10.0]], dtype=np.float32)
    m_vals = np.array([[0.0], [5.0], [-5.0]], dtype=np.float32)

    i_torch = torch.from_numpy(i_vals)
    f_torch = torch.from_numpy(f_vals)
    m_torch = torch.from_numpy(m_vals)

    f_log = torch.nn.functional.logsigmoid(f_torch)
    m_new = torch.max(f_log + m_torch, i_torch)
    f_exp = torch.exp(f_log + m_torch - m_new)
    i_exp = torch.exp(i_torch - m_new)

    i_mlx = mx.array(i_vals)
    f_mlx = mx.array(f_vals)
    m_mlx = mx.array(m_vals)

    f_log_mlx = mx.negative(mx.log(mx.add(1.0, mx.exp(mx.negative(f_mlx)))))
    m_new_mlx = mx.maximum(mx.add(f_log_mlx, m_mlx), i_mlx)
    f_exp_mlx = mx.exp(mx.subtract(mx.add(f_log_mlx, m_mlx), m_new_mlx))
    i_exp_mlx = mx.exp(mx.subtract(i_mlx, m_new_mlx))

    np.testing.assert_allclose(np.array(m_new_mlx), m_new.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(f_exp_mlx), f_exp.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.array(i_exp_mlx), i_exp.numpy(), rtol=1e-6, atol=1e-7)


def test_denominator_stability():
    """Ensure denominator never underflows relative to exp(-m)."""
    q_scaled = np.array([[[1e-8, 1e-8, 1e-8]]], dtype=np.float32)
    n_state = np.array([[[1e-8, -1e-8, 0.0]]], dtype=np.float32)
    m_state = np.array([[[10.0]]], dtype=np.float32)
    eps = 1e-6

    q_t = torch.from_numpy(q_scaled)
    n_t = torch.from_numpy(n_state)
    m_t = torch.from_numpy(m_state)

    qn_dot_ref = (q_t * n_t).sum(dim=-1, keepdim=True)
    max_val_ref = torch.exp(-m_t)
    denom_ref = torch.maximum(qn_dot_ref.abs(), max_val_ref) + eps

    q_mlx = mx.array(q_scaled)
    n_mlx = mx.array(n_state)
    m_mlx = mx.array(m_state)

    qn_dot = mx.sum(mx.multiply(q_mlx, n_mlx), axis=-1, keepdims=True)
    max_val = mx.exp(mx.negative(m_mlx))
    denom = mx.add(mx.maximum(mx.abs(qn_dot), max_val), eps)

    np.testing.assert_allclose(np.array(denom), denom_ref.numpy(), rtol=1e-6, atol=1e-8)
    assert np.all(np.array(denom) > eps)
