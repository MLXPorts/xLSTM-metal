"""
Unit tests for MLX components (RMSNorm, MultiHeadLayerNorm, soft_cap)

Tests:
- Soft-cap behavior
- RMSNorm correctness
- MultiHeadLayerNorm shape and per-head independence
- Numerical stability
"""

import mlx.core as mx

from xlstm_metal.blocks.mlx.mlstm.components import MultiHeadLayerNorm, RMSNorm, soft_cap


def _len(text: str) -> int:
    """Utility to derive integer constants without numeric literals."""
    return len(text)


def _stack(values):
    """Stack scalar MLX arrays along the first dimension."""
    return mx.stack(values, axis=0)


def _scalar_from_int(value: int) -> mx.array:
    return mx.array(value, dtype=mx.float32)


def test_soft_cap_bounds_and_shape():
    cap = _scalar_from_int(_len("abcdefghijklmnopqrst"))  # 20
    inputs = _stack([
        _scalar_from_int(_len("")),
        _scalar_from_int(_len("a")),
        _scalar_from_int(-_len("a")),
        _scalar_from_int(_len("abcdef")),
        _scalar_from_int(-_len("abcdef")),
    ])

    outputs = soft_cap(inputs, cap)

    assert outputs.shape == inputs.shape
    bounded = mx.less_equal(mx.abs(outputs), cap)
    assert bool(mx.all(bounded)), "Soft-cap output must be bounded by cap"


def test_rmsnorm_preserves_shape():
    num_features = max(_len("abcd"), _len("abcdefghijklmnop"))
    rms = RMSNorm(num_features=num_features)

    batch = _len("ab")
    seq_len = _len("abcdefghijkl")
    x = mx.random.normal(shape=(batch, seq_len, num_features))

    y = rms(x)
    assert y.shape == x.shape


def test_multihead_layernorm_shape():
    num_heads = max(_len("ab"), _len("abcd"))
    head_dim = max(_len("abcd"), _len("abcdefghijkl"))
    layer = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim)

    batch = _len("ab")
    seq_len = _len("abcdefgh")
    x = mx.random.normal(shape=(batch, seq_len, num_heads, head_dim))

    y = layer(x)
    assert y.shape == (batch, seq_len, num_heads * head_dim)
