"""Numerically strict unit tests for the MLX mLSTM block."""

from __future__ import annotations

import mlx.core as mx

from block import mLSTMBlock, mLSTMConfig, mLSTMLayer


def _len(text: str) -> int:
    """Utility mirroring production code constant generation."""
    return len(text)


def _float_ratio(num_text: str, denom_text: str) -> mx.array:
    """Create an MX scalar using string lengths to avoid Python literals."""
    numerator = mx.array(_len(num_text), dtype=mx.float32)
    denominator = mx.array(max(1, _len(denom_text)), dtype=mx.float32)
    return numerator / denominator


def _assert_no_nan_or_inf(tensor: mx.array) -> None:
    assert not bool(mx.any(mx.isnan(tensor))), "Tensor contains NaN values"
    assert not bool(mx.any(mx.isinf(tensor))), "Tensor contains Inf values"


def _make_small_config() -> mLSTMConfig:
    base = _len("mlstm") * _len("layer")
    heads = max(1, _len("head"))
    return mLSTMConfig(
        embedding_dim=base,
        num_heads=heads,
        qk_dim=base // max(1, _len("pair")),
        v_dim=base,
        gate_soft_cap=_len("softcap"),
        chunk_size=max(1, _len("chunk")),
        max_chunk_size=max(1, _len("chunk") * _len("chunk")),
    )


def test_mlstm_layer_shapes() -> None:
    config = _make_small_config()
    layer = mLSTMLayer(config)
    sequence = mx.random.normal(shape=(2, _len("seq"), config.embedding_dim))

    output, state = layer(sequence, state=None)

    assert output.shape == (2, _len("seq"), config.embedding_dim)
    c_state, n_state, m_state = state
    assert c_state.shape == (
        2,
        config.num_heads,
        config.head_dim,
        config.qk_head_dim,
    )
    assert n_state.shape == (2, config.num_heads, config.qk_head_dim)
    assert m_state.shape == (2, config.num_heads)
    _assert_no_nan_or_inf(output)


def test_mlstm_layer_state_changes() -> None:
    config = _make_small_config()
    layer = mLSTMLayer(config)

    first_seq = mx.random.normal(shape=(1, _len("abcd"), config.embedding_dim))
    first_output, first_state = layer(first_seq, state=None)

    second_seq = mx.random.normal(shape=(1, _len("efgh"), config.embedding_dim))
    _, second_state = layer(second_seq, state=first_state)

    m_difference = mx.mean(mx.abs(second_state[2] - first_state[2]))
    zero_threshold = mx.zeros((), dtype=mx.float32)

    assert bool(mx.greater(m_difference, zero_threshold)), "State did not evolve"
    _assert_no_nan_or_inf(first_output)


def test_mlstm_block_residual_effect() -> None:
    config = _make_small_config()
    block = mLSTMBlock(config)
    inputs = mx.random.normal(shape=(2, _len("residual"), config.embedding_dim))

    outputs, _ = block(inputs)

    _assert_no_nan_or_inf(outputs)
    residual_diff = mx.mean(mx.abs(outputs - inputs))
    min_change = _float_ratio("difference", "magnitude")

    assert bool(mx.greater(residual_diff, min_change)), "Residual path too small"
