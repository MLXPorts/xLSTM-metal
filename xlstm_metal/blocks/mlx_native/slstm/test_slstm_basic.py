"""
Basic test for sLSTM implementation

Verifies that sLSTM block can be instantiated and runs forward pass.
"""

import sys
from pathlib import Path

import mlx.core as mx

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from xlstm_metal.blocks.slstm_mlx import sLSTMBlock, sLSTMConfig


def test_slstm_instantiation():
    """Test that sLSTM block can be created"""
    print("\n" + "=" * 80)
    print("Test 1: sLSTM Block Instantiation")
    print("=" * 80)

    config = sLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        gate_soft_cap=15.0
    )

    block = sLSTMBlock(config)

    print(f"✓ Created sLSTMBlock with config:")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  head_dim: {config.head_dim}")
    print()


def test_slstm_forward_pass():
    """Test forward pass through sLSTM"""
    print("=" * 80)
    print("Test 2: sLSTM Forward Pass (stateless)")
    print("=" * 80)

    config = sLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        gate_soft_cap=15.0
    )

    block = sLSTMBlock(config)

    # Create test input
    batch_size = 2
    seq_len = 8
    x = mx.random.normal((batch_size, seq_len, config.embedding_dim))

    print(f"Input shape: {x.shape}")

    # Forward pass (no state)
    output, state = block(x, state=None)

    print(f"Output shape: {output.shape}")
    print(f"State returned: {state is not None}")

    if state is not None:
        c_state, n_state, m_state = state
        print(f"  c_state shape: {c_state.shape}")
        print(f"  n_state shape: {n_state.shape}")
        print(f"  m_state shape: {m_state.shape}")

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print("\n✓ Forward pass successful\n")


def test_slstm_stateful():
    """Test stateful forward pass"""
    print("=" * 80)
    print("Test 3: sLSTM Stateful Forward Pass")
    print("=" * 80)

    config = sLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        gate_soft_cap=15.0,
        return_last_states=True
    )

    block = sLSTMBlock(config)

    # First forward pass
    x1 = mx.random.normal((1, 4, config.embedding_dim))
    output1, state1 = block(x1, state=None)

    print(f"First pass - Input: {x1.shape}, Output: {output1.shape}")

    # Second forward pass with state
    x2 = mx.random.normal((1, 4, config.embedding_dim))
    output2, state2 = block(x2, state=state1)

    print(f"Second pass - Input: {x2.shape}, Output: {output2.shape}")
    print(f"State preserved: {state2 is not None}")

    print("\n✓ Stateful forward pass successful\n")


def test_slstm_kernel_only():
    """Test sLSTM kernel directly"""
    print("=" * 80)
    print("Test 4: sLSTM Kernel Direct Test")
    print("=" * 80)

    from xlstm_metal.blocks.slstm_mlx.kernel import slstm_recurrent_step

    B, NH, H = 2, 4, 128

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

    print(f"Input shapes: z={z.shape}, gates={i_preact.shape}")
    print(f"Output shapes: h={h.shape}, c={c_new.shape}, n={n_new.shape}, m={m_new.shape}")

    assert h.shape == (B, NH, H), f"h shape mismatch: {h.shape}"
    assert c_new.shape == (B, NH, H), f"c shape mismatch: {c_new.shape}"
    assert n_new.shape == (B, NH, H), f"n shape mismatch: {n_new.shape}"
    assert m_new.shape == (B, NH), f"m shape mismatch: {m_new.shape}"

    print("\n✓ Kernel test successful\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("sLSTM MLX Implementation - Basic Tests")
    print("=" * 80)

    try:
        test_slstm_instantiation()
        test_slstm_forward_pass()
        test_slstm_stateful()
        test_slstm_kernel_only()

        print("\n" + "=" * 80)
        print("All Tests Passed! ✓")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
