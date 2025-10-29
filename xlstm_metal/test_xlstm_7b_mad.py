#!/usr/bin/env python
"""
Test xLSTM-7B MAD Integration

Verifies that the MAD wiring system can instantiate and run xLSTM-7B.
"""

import mlx.core as mx
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xlstm_metal.blocks.mlx.wiring import create_xlstm_7b_wiring, WiredMADModel


def test_wiring_creation():
    """Test that we can create xLSTM-7B wiring."""
    print("\n" + "=" * 80)
    print("Test 1: Create xLSTM-7B Wiring")
    print("=" * 80)

    wiring = create_xlstm_7b_wiring(
        embedding_dim=4096,
        num_heads=8,
        num_blocks=32,
        vocab_size=50304
    )

    print(f"✓ Created wiring with {wiring.num_blocks} blocks")
    print(f"  Block names: {wiring.block_names[:5]}... (showing first 5)")

    # Check execution stages
    stages = wiring.get_execution_stages()
    print(f"  Execution stages: {len(stages)}")
    print(f"  Stage 0: {stages[0]}")
    print(f"  Stage 1: {stages[1]}")
    print(f"  Stage -1: {stages[-1]}")

    assert len(stages) == 35, f"Expected 35 stages, got {len(stages)}"
    assert stages[0] == ['embedding'], "First stage should be embedding"
    assert stages[-1] == ['lm_head'], "Last stage should be lm_head"

    print("\n✓ Wiring creation successful\n")
    return wiring


def test_model_instantiation(wiring):
    """Test that we can instantiate WiredMADModel."""
    print("\n" + "=" * 80)
    print("Test 2: Instantiate WiredMADModel")
    print("=" * 80)

    model = WiredMADModel(
        wiring=wiring,
        input_block='embedding',
        output_block='lm_head'
    )

    print(f"✓ Created WiredMADModel")
    print(f"  Total blocks: {len(model.blocks)}")
    print(f"  Execution stages: {len(model.stages)}")

    # Check that blocks are lazily instantiated
    print(f"\n  Instantiated blocks: {list(model.blocks.keys())[:5]}... (showing first 5)")

    print("\n✓ Model instantiation successful\n")
    return model


def test_forward_pass(model):
    """Test forward pass through the model."""
    print("\n" + "=" * 80)
    print("Test 3: Forward Pass")
    print("=" * 80)

    batch_size = 2
    seq_len = 8
    input_ids = mx.random.randint(0, 50304, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    logits, state = model(input_ids, hidden_states=None)

    print(f"Output logits shape: {logits.shape}")
    print(f"State keys: {list(state.keys())[:5]}... (showing first 5)")

    # Verify shapes
    assert logits.shape == (batch_size, seq_len, 50304), \
        f"Expected logits shape {(batch_size, seq_len, 50304)}, got {logits.shape}"

    # Check that we have state for xLSTM blocks
    num_xlstm_states = sum(1 for k in state.keys() if k.startswith('xlstm_'))
    print(f"  xLSTM blocks with state: {num_xlstm_states}")
    assert num_xlstm_states == 32, f"Expected 32 xLSTM states, got {num_xlstm_states}"

    print("\n✓ Forward pass successful\n")


def test_stateful_generation(model):
    """Test stateful generation (one token at a time)."""
    print("\n" + "=" * 80)
    print("Test 4: Stateful Generation")
    print("=" * 80)

    # Initial prompt
    prompt = mx.array([[1, 2, 3, 4, 5]])  # [1, 5]
    print(f"Prompt shape: {prompt.shape}")

    # First forward pass
    logits1, state1 = model(prompt, hidden_states=None)
    print(f"Initial logits shape: {logits1.shape}")
    print(f"Initial state keys: {len(state1)}")

    # Generate next token
    next_token = mx.argmax(logits1[0, -1, :])
    print(f"Next token: {int(next_token)}")

    # Second forward pass (stateful - only process new token)
    next_input = mx.array([[int(next_token)]])  # [1, 1]
    logits2, state2 = model(next_input, hidden_states=state1)

    print(f"Second logits shape: {logits2.shape}")
    print(f"Updated state keys: {len(state2)}")

    assert logits2.shape == (1, 1, 50304), \
        f"Expected logits shape (1, 1, 50304), got {logits2.shape}"

    print("\n✓ Stateful generation successful\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("xLSTM-7B MAD Integration Tests")
    print("=" * 80)

    wiring = test_wiring_creation()
    model = test_model_instantiation(wiring)
    test_forward_pass(model)
    test_stateful_generation(model)

    print("\n" + "=" * 80)
    print("All Tests Passed! ✓")
    print("=" * 80 + "\n")
