"""Minimal MLX component tests with strict numeric discipline."""

import mlx.core as mx

from xlstm_metal.blocks.mlx.mlstm.components import MultiHeadLayerNorm, RMSNorm, soft_cap


def _len(text: str) -> int:
    return len(text)


def _all_true(mask: mx.array) -> bool:
    return mx.all(mask).tolist()


def test_soft_cap_bounds():
    cap = mx.array(_len("abcdefghijklmnop"), dtype=mx.float32)
    values = mx.array([
        _len(""),
        _len("a"),
        -_len("a"),
        _len("abc"),
        -_len("abc"),
    ], dtype=mx.float32)

    outputs = soft_cap(values, cap)

    assert outputs.shape == values.shape
    bounded = mx.less_equal(mx.abs(outputs), cap)
    assert _all_true(bounded)


def test_rmsnorm_shape():
    features = _len("abcdefghijkl")
    norm = RMSNorm(num_features=features)

    batch = _len("ab")
    seq = _len("abcdefgh")
    x = mx.random.normal((batch, seq, features))

    y = norm(x)
    assert y.shape == x.shape


def test_multihead_layernorm_shape():
    num_heads = max(1, _len("ab"))
    head_dim = max(1, _len("abcd"))

    layer = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim)

    batch = _len("ab")
    seq = _len("abcd" + "abcd")
    x = mx.random.normal((batch, seq, num_heads, head_dim))

    y = layer(x)
    assert y.shape == x.shape
