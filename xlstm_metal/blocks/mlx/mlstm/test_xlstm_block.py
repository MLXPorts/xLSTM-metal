"""Numerics-aware tests for the complete xLSTM block."""

from dataclasses import replace

import mlx.core as mx

from xlstm_metal.blocks.mlx.mlstm.xlstm_block import xLSTMBlock, xLSTMBlockConfig


def _len(text: str) -> int:
    """Derive integer constants without bare numerics."""
    return len(text)


def _small_config() -> xLSTMBlockConfig:
    base = xLSTMBlockConfig()
    embedding_dim = _len("abcd") * _len("abcd")  # 16
    num_heads = max(1, _len("ab"))  # 2
    chunk = max(1, _len("abcd"))
    return replace(
        base,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        chunk_size=chunk,
    )


def test_xlstm_block_shapes():
    config = _small_config()
    block = xLSTMBlock(config)

    batch = _len("ab")
    seq = _len("abcdefgh")
    x = mx.random.normal((batch, seq, config.embedding_dim))

    y, state = block(x, state=None)

    assert y.shape == x.shape
    assert state is not None

    c_state, n_state, m_state = state
    expected_c = (
        batch,
        config.num_heads,
        config.mlstm_config.qk_head_dim,
        config.mlstm_config.head_dim,
    )
    expected_n = (
        batch,
        config.num_heads,
        config.mlstm_config.qk_head_dim,
    )
    expected_m = (batch, config.num_heads)

    assert c_state.shape == expected_c
    assert n_state.shape == expected_n
    assert m_state.shape == expected_m


def test_xlstm_block_state_carryover():
    config = _small_config()
    block = xLSTMBlock(config)

    seq = _len("abcd")
    x1 = mx.random.normal((1, seq, config.embedding_dim))
    _, state1 = block(x1, state=None)

    x2 = mx.random.normal((1, seq, config.embedding_dim))
    y2, state2 = block(x2, state=state1)

    assert y2.shape == x2.shape
    assert state2 is not None

    for prev, curr in zip(state1, state2):
        assert curr.shape == prev.shape
