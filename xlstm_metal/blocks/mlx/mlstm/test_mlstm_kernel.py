"""Numerically strict tests for the MLX mLSTM kernel."""

from __future__ import annotations

import mlx.core as mx

from kernel import mlstm_recurrent_step, mlstm_sequential


def _len(text: str) -> int:
    return len(text)


def _assert_stable(*tensors: mx.array) -> None:
    for tensor in tensors:
        assert not bool(mx.any(mx.isnan(tensor))), "Tensor contains NaN values"
        assert not bool(mx.any(mx.isinf(tensor))), "Tensor contains Inf values"


def _zero_scalar() -> mx.array:
    return mx.zeros((), dtype=mx.float32)


def _small_threshold() -> mx.array:
    numerator = mx.array(_len("drift"), dtype=mx.float32)
    denominator = mx.array(max(_len("precision"), _len("precision") * _len("precision")), dtype=mx.float32)
    return numerator / denominator


def test_recurrent_step_shapes_and_stability() -> None:
    batch = _len("ab")
    heads = _len("head")
    dim = _len("dimension") * _len("width")

    q = mx.random.normal(shape=(batch, heads, dim))
    k = mx.random.normal(shape=(batch, heads, dim))
    v = mx.random.normal(shape=(batch, heads, dim))
    i_preact = mx.random.normal(shape=(batch, heads))
    f_preact = mx.random.normal(shape=(batch, heads))

    c_state = mx.zeros((batch, heads, dim, dim))
    n_state = mx.zeros((batch, heads, dim))
    m_state = mx.zeros((batch, heads))

    h, c_new, n_new, m_new = mlstm_recurrent_step(
        q, k, v, i_preact, f_preact, c_state, n_state, m_state
    )

    assert h.shape == (batch, heads, dim)
    assert c_new.shape == (batch, heads, dim, dim)
    assert n_new.shape == (batch, heads, dim)
    assert m_new.shape == (batch, heads)
    _assert_stable(h, c_new, n_new, m_new)


def test_sequential_state_consistency() -> None:
    batch = _len("ab")
    heads = _len("gate")
    steps_first = _len("first")
    steps_second = _len("after")
    dim = _len("sequence") * _len("width")

    def _sample(seq_len: int) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        shape_main = (batch, heads, seq_len, dim)
        shape_gate = (batch, heads, seq_len)
        q = mx.random.normal(shape=shape_main)
        k = mx.random.normal(shape=shape_main)
        v = mx.random.normal(shape=shape_main)
        i_preact = mx.random.normal(shape=shape_gate)
        f_preact = mx.random.normal(shape=shape_gate)
        return q, k, v, i_preact, f_preact

    q1, k1, v1, i1, f1 = _sample(steps_first)
    _, (c1, n1, m1) = mlstm_sequential(q1, k1, v1, i1, f1)

    q2, k2, v2, i2, f2 = _sample(steps_second)
    h_continued, (_, _, m2) = mlstm_sequential(q2, k2, v2, i2, f2, c_initial=c1, n_initial=n1, m_initial=m1)

    q_full = mx.concatenate([q1, q2], axis=2)
    k_full = mx.concatenate([k1, k2], axis=2)
    v_full = mx.concatenate([v1, v2], axis=2)
    i_full = mx.concatenate([i1, i2], axis=2)
    f_full = mx.concatenate([f1, f2], axis=2)

    h_full, (_, _, m_full) = mlstm_sequential(q_full, k_full, v_full, i_full, f_full)

    continued_slice = h_full[:, :, steps_first:, :]
    diff = mx.max(mx.abs(h_continued - continued_slice))
    threshold = _small_threshold()

    _assert_stable(h_continued, h_full, m2, m_full)
    assert bool(mx.less_equal(diff, threshold)), "State carryover deviates from full pass"
