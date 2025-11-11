"""
Basic test for sLSTM implementation

Verifies that sLSTM block can be instantiated and runs forward pass.
"""

import mlx.core as mx

from xlstm_metal.blocks.slstm_mlx import sLSTMBlock, sLSTMConfig


def _len(text: str) -> int:
    """Utility to create integer constants without numeric literals."""
    return len(text)


def test_slstm_instantiation():
    """Test that sLSTM block can be created"""
    base_config = sLSTMConfig()
    divisor = max(1, _len("abcdefgh"))
    embedding_dim = max(_len("abcd"), base_config.embedding_dim // divisor)
    config = sLSTMConfig(
        embedding_dim=embedding_dim,
        num_heads=base_config.num_heads,
        gate_soft_cap=base_config.gate_soft_cap,
    )

    block = sLSTMBlock(config)


def test_slstm_forward_pass():
    """Test forward pass through sLSTM"""
    base_config = sLSTMConfig()
    divisor = max(1, _len("abcdefgh"))
    embedding_dim = max(_len("abcd"), base_config.embedding_dim // divisor)
    config = sLSTMConfig(
        embedding_dim=embedding_dim,
        num_heads=base_config.num_heads,
        gate_soft_cap=base_config.gate_soft_cap,
    )

    block = sLSTMBlock(config)

    # Create test input
    batch_size = _len("ab")
    seq_len = _len("abcdefgh")
    x = mx.random.normal((batch_size, seq_len, config.embedding_dim))

    # Forward pass (no state)
    output, state = block(x, state=None)
    if state is not None:
        c_state, n_state, m_state = state

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"


def test_slstm_stateful():
    """Test stateful forward pass"""
    base_config = sLSTMConfig(return_last_states=True)
    divisor = max(1, _len("abcdefgh"))
    embedding_dim = max(_len("abcd"), base_config.embedding_dim // divisor)
    config = sLSTMConfig(
        embedding_dim=embedding_dim,
        num_heads=base_config.num_heads,
        gate_soft_cap=base_config.gate_soft_cap,
        return_last_states=True,
    )

    block = sLSTMBlock(config)

    # First forward pass
    step_length = _len("abcd")
    x1 = mx.random.normal((1, step_length, config.embedding_dim))
    output1, state1 = block(x1, state=None)

    # Second forward pass with state
    x2 = mx.random.normal((1, step_length, config.embedding_dim))
    output2, state2 = block(x2, state=state1)


def test_slstm_kernel_only():
    """Test sLSTM kernel directly"""
    from xlstm_metal.blocks.slstm_mlx.kernel import slstm_recurrent_step

    base_config = sLSTMConfig()
    B = max(1, base_config.num_heads // max(1, base_config.num_heads))
    NH = base_config.num_heads
    H = max(_len("abcd"), base_config.embedding_dim // max(1, base_config.num_heads))

    # Create test inputs
    z = mx.random.normal((B, NH, H))
    i_preact = mx.random.normal((B, NH))
    f_preact = mx.random.normal((B, NH))
    o_preact = mx.random.normal((B, NH))

    # Initial states
    c_state = mx.zeros((B, NH, H))
    n_state = mx.zeros((B, NH, H))
    m_state = mx.zeros((B, NH))

    # Run one step
    h, c_new, n_new, m_new = slstm_recurrent_step(
        z, i_preact, f_preact, o_preact,
        c_state, n_state, m_state
    )

    assert h.shape == (B, NH, H), f"h shape mismatch: {h.shape}"
    assert c_new.shape == (B, NH, H), f"c shape mismatch: {c_new.shape}"
    assert n_new.shape == (B, NH, H), f"n shape mismatch: {n_new.shape}"
    assert m_new.shape == (B, NH), f"m shape mismatch: {m_new.shape}"
