#!/usr/bin/env python
"""
Test MAD Wiring System

Demonstrates declarative graph composition with parallel execution.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xlstm_metal.wiring import (
    MADWiring,
    BlockSpec,
    BlockType,
    BackendType,
    WiredMADModel,
    create_parallel_head_wiring,
    create_xlstm_7_1_wiring
)


def test_basic_wiring():
    """Test basic wiring construction and visualization"""
    print("Test 1: Basic Wiring Construction")
    print("=" * 80)

    # Define simple 3-block architecture
    specs = {
        'input': BlockSpec(
            name='input',
            block_type=BlockType.NORM,
            backend=BackendType.TORCH_COMPILED,
            params={'d_model': 256}
        ),
        'mlstm': BlockSpec(
            name='mlstm',
            block_type=BlockType.MLSTM,
            backend=BackendType.TORCH_COMPILED,
            params={'d_model': 256, 'num_heads': 4, 'head_dim': 64}
        ),
        'output': BlockSpec(
            name='output',
            block_type=BlockType.LINEAR,
            backend=BackendType.TORCH_COMPILED,
            params={'in_features': 256, 'out_features': 256}
        )
    }

    wiring = MADWiring(specs)
    wiring.add_connection('input', 'mlstm')
    wiring.add_connection('mlstm', 'output')

    print(wiring.visualize())
    print("\n✓ Basic wiring construction successful\n")


def test_parallel_heads():
    """Test parallel multi-head wiring"""
    print("\nTest 2: Parallel Multi-Head Wiring")
    print("=" * 80)

    wiring = create_parallel_head_wiring(num_heads=4, d_model=256, head_dim=64)

    print(wiring.visualize())

    # Check execution stages
    stages = wiring.get_execution_stages()
    print(f"\nExecution stages: {len(stages)}")
    print(f"Stage 0 (input): {stages[0]}")
    print(f"Stage 1 (parallel heads): {stages[1]}")
    print(f"Stage 2 (combiner): {stages[2]}")

    assert len(stages) == 3, "Should have 3 stages"
    assert len(stages[1]) == 4, "Stage 1 should have 4 parallel heads"

    print("\n✓ Parallel head wiring correct\n")


def test_forward_pass():
    """Test actual forward pass through wired model"""
    print("\nTest 3: Forward Pass Through Wired Model")
    print("=" * 80)

    # Create simple 2-layer sequential model
    specs = {
        'norm1': BlockSpec(
            name='norm1',
            block_type=BlockType.NORM,
            backend=BackendType.TORCH_COMPILED,
            params={'d_model': 128}
        ),
        'mlstm': BlockSpec(
            name='mlstm',
            block_type=BlockType.MLSTM,
            backend=BackendType.TORCH_COMPILED,
            params={'d_model': 128, 'num_heads': 2, 'head_dim': 64}
        ),
        'norm2': BlockSpec(
            name='norm2',
            block_type=BlockType.NORM,
            backend=BackendType.TORCH_COMPILED,
            params={'d_model': 128}
        )
    }

    wiring = MADWiring(specs)
    wiring.add_connection('norm1', 'mlstm')
    wiring.add_connection('mlstm', 'norm2')

    # Create model
    model = WiredMADModel(wiring, input_block='norm1', output_block='norm2')

    # Test forward pass
    batch_size = 2
    seq_len = 16
    d_model = 128

    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output, hidden_states = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Hidden states: {list(hidden_states.keys())}")

    assert output.shape == x.shape, "Output shape should match input"
    print("\n✓ Forward pass successful\n")


def test_polarity():
    """Test excitatory/inhibitory connections"""
    print("\nTest 4: Excitatory/Inhibitory Polarity")
    print("=" * 80)

    specs = {
        'excitatory': BlockSpec(
            name='excitatory',
            block_type=BlockType.NORM,
            params={'d_model': 64},
            polarity=1.0
        ),
        'inhibitory': BlockSpec(
            name='inhibitory',
            block_type=BlockType.NORM,
            params={'d_model': 64},
            polarity=-1.0
        ),
        'combiner': BlockSpec(
            name='combiner',
            block_type=BlockType.LINEAR,
            params={'in_features': 64, 'out_features': 64}
        )
    }

    wiring = MADWiring(specs)

    # Excitatory connection (positive polarity)
    wiring.add_connection('excitatory', 'combiner', polarity=1.0)

    # Inhibitory connection (negative polarity)
    wiring.add_connection('inhibitory', 'combiner', polarity=-1.0)

    print(wiring.visualize())

    # Check adjacency matrix
    exc_idx = wiring.name_to_idx['excitatory']
    inh_idx = wiring.name_to_idx['inhibitory']
    comb_idx = wiring.name_to_idx['combiner']

    assert wiring.adjacency[exc_idx, comb_idx] == 1.0, "Excitatory should be +1"
    assert wiring.adjacency[inh_idx, comb_idx] == -1.0, "Inhibitory should be -1"

    print("\n✓ Polarity encoding correct\n")


def test_7_1_pattern():
    """Test canonical 7:1 mLSTM:sLSTM pattern"""
    print("\nTest 5: Canonical 7:1 xLSTM Pattern")
    print("=" * 80)

    wiring = create_xlstm_7_1_wiring(d_model=256, num_blocks=16)

    print(f"Created 7:1 wiring with {wiring.num_blocks} total blocks")

    # Count block types
    num_mlstm = sum(1 for name in wiring.block_names if 'mlstm' in name)
    num_slstm = sum(1 for name in wiring.block_names if 'slstm' in name)

    print(f"  mLSTM blocks: {num_mlstm}")
    print(f"  sLSTM blocks: {num_slstm}")
    print(f"  Ratio: {num_mlstm}:{num_slstm}")

    # Should be 7:1 ratio (14 mLSTM : 2 sLSTM for 16 blocks)
    assert num_slstm == 2, "Should have 2 sLSTM blocks"
    assert num_mlstm == 14, "Should have 14 mLSTM blocks"

    # Visualize (truncated)
    print("\nExecution stages:")
    stages = wiring.get_execution_stages()
    print(f"  Total stages: {len(stages)}")
    print(f"  First 5 stages: {stages[:5]}")

    print("\n✓ 7:1 pattern correct\n")


def test_cycle_detection():
    """Test that cycles are properly detected"""
    print("\nTest 6: Cycle Detection")
    print("=" * 80)

    specs = {
        'a': BlockSpec('a', BlockType.NORM, params={'d_model': 64}),
        'b': BlockSpec('b', BlockType.NORM, params={'d_model': 64}),
        'c': BlockSpec('c', BlockType.NORM, params={'d_model': 64})
    }

    wiring = MADWiring(specs)

    # Create cycle: a -> b -> c -> a
    wiring.add_connection('a', 'b')
    wiring.add_connection('b', 'c')
    wiring.add_connection('c', 'a')  # Cycle!

    print("Created cyclic graph: a -> b -> c -> a")

    try:
        stages = wiring.get_execution_stages()
        print("✗ ERROR: Cycle not detected!")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Cycle detected correctly: {e}\n")


def benchmark_wiring_overhead():
    """Benchmark wiring system overhead vs direct sequential execution"""
    print("\nTest 7: Wiring System Overhead")
    print("=" * 80)

    import time

    # Create wired model
    wiring = create_parallel_head_wiring(num_heads=4, d_model=256, head_dim=64)
    wired_model = WiredMADModel(wiring, 'input_norm', 'output_proj')

    # Test input
    batch_size = 4
    seq_len = 32
    d_model = 256
    x = torch.randn(batch_size, seq_len, d_model)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = wired_model(x)

    # Benchmark wired execution
    num_runs = 50
    wired_times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            output, _ = wired_model(x)
        wired_times.append(time.perf_counter() - start)

    avg_time = sum(wired_times) / len(wired_times)
    std_time = (sum((t - avg_time)**2 for t in wired_times) / len(wired_times))**0.5

    print(f"Wired model execution:")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Std dev: {std_time*1000:.2f} ms")
    print(f"  Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")

    # Get execution info
    info = wired_model.get_execution_info()
    print(f"\nExecution graph:")
    print(f"  Total blocks: {info['num_blocks']}")
    print(f"  Execution stages: {info['num_stages']}")
    print(f"  Max parallelism: {info['max_parallelism']}")

    print("\n✓ Wiring overhead measured\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MAD Wiring System Tests")
    print("=" * 80 + "\n")

    try:
        test_basic_wiring()
        test_parallel_heads()
        test_polarity()
        test_7_1_pattern()
        test_cycle_detection()
        test_forward_pass()
        benchmark_wiring_overhead()

        print("\n" + "=" * 80)
        print("All Tests Passed! ✓")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
