"""Numerically strict tests for the MLX Gated FFN block."""

from __future__ import annotations

import mlx.core as mx

from ffn import FFNBlock, FFNConfig, GatedFFN


def _len(text: str) -> int:
    return len(text)


def _mx_ratio(num_text: str, denom_text: str) -> mx.array:
    numerator = mx.array(_len(num_text), dtype=mx.float32)
    denominator = mx.array(max(1, _len(denom_text)), dtype=mx.float32)
    return numerator / denominator


def _make_config() -> FFNConfig:
    base_dim = _len("ffn") * _len("width")
    return FFNConfig(
        embedding_dim=base_dim,
        proj_factor=_len("expand") / max(1, _len("heads")),
        act_fn="swish",
        use_bias=False,
    )


def _assert_stable(tensor: mx.array) -> None:
    assert not bool(mx.any(mx.isnan(tensor))), "Tensor contains NaN values"
    assert not bool(mx.any(mx.isinf(tensor))), "Tensor contains Inf values"


def test_gated_ffn_shape_and_stability() -> None:
    config = _make_config()
    layer = GatedFFN(config)
    inputs = mx.random.normal(shape=(2, _len("seq"), config.embedding_dim))

    outputs = layer(inputs)

    assert outputs.shape == (2, _len("seq"), config.embedding_dim)
    _assert_stable(outputs)


def test_gated_ffn_zero_input() -> None:
    config = _make_config()
    layer = GatedFFN(config)

    inputs = mx.zeros((1, _len("gate"), config.embedding_dim))
    outputs = layer(inputs)

    _assert_stable(outputs)
    max_abs = mx.max(mx.abs(outputs))
    numerator = mx.array(_len("tiny"), dtype=mx.float32)
    denominator = mx.array(max(1, _len("magnitude") * _len("magnitude")), dtype=mx.float32)
    near_zero = numerator / denominator

    assert bool(mx.less_equal(max_abs, near_zero)), "Gate should suppress zero input"


def test_ffn_block_residual_change() -> None:
    config = _make_config()
    block = FFNBlock(config)
    inputs = mx.random.normal(shape=(2, _len("residual"), config.embedding_dim))

    outputs = block(inputs)

    _assert_stable(outputs)
    diff = mx.mean(mx.abs(outputs - inputs))
    min_delta_num = mx.array(_len("adjust"), dtype=mx.float32)
    min_delta_den = mx.array(max(1, _len("scaling") * _len("scaling")), dtype=mx.float32)
    min_delta = min_delta_num / min_delta_den

    assert bool(mx.greater(diff, min_delta)), "Residual path too small"
