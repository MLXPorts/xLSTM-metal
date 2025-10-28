#!/usr/bin/env python
"""
Test xLSTM NCPS Wiring

Verifies that the NCPS wiring system works correctly for xLSTM.
"""

import mlx.core as mx
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xlstm_metal.mlx_blocks.wiring import create_xlstm_wiring


def test_wiring_creation():
    """Test that we can create xLSTM-7B wiring."""
    print("\n" + "=" * 80)
    print("Test 1: Create xLSTM-7B Wiring")
    print("=" * 80)

    # Use config-driven wiring
    config = {
        'embedding_dim': 4096,
        'num_heads': 8,
        'num_blocks': 32,
        'vocab_size': 50304,
        'qk_dim_factor': 0.5,
        'v_dim_factor': 1.0,
        'ffn_proj_factor': 2.671875,
        'gate_soft_cap': 15.0,
        'norm_eps': 1e-6,
        'output_logit_soft_cap': 30.0
    }

    wiring = create_xlstm_wiring(config)

    print(f"✓ Created wiring with {wiring.units} neurons (xLSTM blocks)")
    print(f"  Wiring type: {type(wiring).__name__}")
    print(f"  Number of layers: {wiring.num_layers}")

    # For xLSTMWiring, it's a simple sequential wiring
    # Units = number of xLSTM blocks
    assert wiring.units == config['num_blocks'], \
        f"Expected {config['num_blocks']} units, got {wiring.units}"

    # Get neurons of first layer
    layer_0_neurons = wiring.get_neurons_of_layer(0)
    print(f"  Layer 0 neurons: {len(layer_0_neurons)} blocks")
    assert len(layer_0_neurons) == config['num_blocks'], \
        f"Expected {config['num_blocks']} neurons in layer 0"

    print("\n✓ Wiring creation successful\n")
    return wiring


def test_wiring_properties(wiring):
    """Test wiring properties."""
    print("\n" + "=" * 80)
    print("Test 2: Wiring Properties")
    print("=" * 80)

    # Test NCPS Wiring properties
    print(f"✓ Wiring properties:")
    print(f"  units (neurons): {wiring.units}")
    print(f"  num_layers: {wiring.num_layers}")
    print(f"  is_built: {wiring.is_built()}")

    # Visualize the wiring
    print(f"\n  Wiring visualization (first 500 chars):")
    viz = wiring.visualize()
    print(viz[:500] + "..." if len(viz) > 500 else viz)

    print("\n✓ Wiring properties test successful\n")
    return wiring


def test_basic_functionality(wiring):
    """Test basic wiring functionality."""
    print("\n" + "=" * 80)
    print("Test 3: Basic Wiring Functionality")
    print("=" * 80)

    # Build wiring with input dimension
    input_dim = 4096
    wiring.build(input_dim)

    print(f"✓ Built wiring with input_dim={input_dim}")
    print(f"  input_dim: {wiring.input_dim}")
    print(f"  output_dim: {wiring.output_dim}")

    # Test neuron type queries
    neuron_0_type = wiring.get_type_of_neuron(0)
    print(f"  Neuron 0 type: {neuron_0_type}")

    # Test layer queries
    layer_0_neurons = wiring.get_neurons_of_layer(0)
    print(f"  Layer 0 has {len(layer_0_neurons)} neurons")

    print("\n✓ Basic functionality test successful\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("xLSTM-7B NCPS Wiring Tests")
    print("=" * 80)

    wiring = test_wiring_creation()
    wiring = test_wiring_properties(wiring)
    test_basic_functionality(wiring)

    print("\n" + "=" * 80)
    print("All Tests Passed! ✓")
    print("=" * 80 + "\n")

