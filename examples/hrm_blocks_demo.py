#!/usr/bin/env python
"""
Simple HRM+ Blocks Demo

Tests individual HRM+ components as proper MAD blocks.
"""

import mlx.core as mx
from xlstm_metal.blocks.hrm_mlx import (
    MemoryCubeMLX,
    CubeGatedBlockMLX,
    ACTHaltingHeadMLX,
    LiquidTimeConstantMLX,
    z5_slots,
    boundary_commit_mask
)


def demo_memory_cube():
    """Test Memory Cube retrieval and updates."""
    print("=" * 80)
    print("Memory Cube Demo")
    print("=" * 80)

    cube = MemoryCubeMLX(d_key=64, d_val=64, max_items=100, topk=5)

    # Add some memories
    print("\nAdding 10 memories to cube...")
    keys = mx.random.normal((10, 64))
    vals = mx.random.normal((10, 64))
    cube.update(keys, vals)

    print(f"Cube now has {cube.keys.shape[0]} items")

    # Query
    print("\nQuerying with 3 new keys...")
    queries = mx.random.normal((3, 64))
    preds, confs = cube.query(queries)

    print(f"Predictions shape: {preds.shape}")
    print(f"Confidences: {confs}")
    print(f"Mean confidence: {mx.mean(confs).item():.3f}")

    print("\n✓ Memory Cube working!")


def demo_cube_gated_block():
    """Test Cube-Gated Block with MAD interface."""
    print("\n" + "=" * 80)
    print("Cube-Gated Block Demo (MAD Interface)")
    print("=" * 80)

    block = CubeGatedBlockMLX(
        d_in=128,
        fuse_phase_keys=True,
        k_5ht=0.5,
        max_items=50,
        topk=5
    )

    # Test input
    batch_size, seq_len, d_model = 2, 10, 128
    h_in = mx.random.normal((batch_size, seq_len, d_model))

    # Generate times for Z5 scheduler
    times = mx.broadcast_to(mx.arange(seq_len).reshape(1, -1), (batch_size, seq_len))

    print(f"\nInput shape: {h_in.shape}")
    print(f"Times (Z5 slots): {z5_slots(times[0]).tolist()}")
    print(f"Boundary commits: {boundary_commit_mask(times[0]).tolist()}")

    # Forward pass with MAD interface
    print("\nForward pass (inference mode)...")
    state = {'times': times, 'train': False}
    output, new_state = block(h_in, state)

    print(f"Output shape: {output.shape}")
    print(f"Alpha (gate strength): {new_state['alpha_mean']:.3f}")
    print(f"Confidence: {new_state['conf_mean']:.3f}")
    print(f"Energy ratio: {new_state['energy_post_gate'] / new_state['energy_pre_gate']:.3f}")

    print("\n✓ Cube-Gated Block working with MAD interface!")


def demo_act_halting():
    """Test ACT Halting Head."""
    print("\n" + "=" * 80)
    print("ACT Halting Head Demo")
    print("=" * 80)

    head = ACTHaltingHeadMLX(d_model=128, threshold=0.5)

    # Test input
    h = mx.random.normal((2, 10, 128))

    print(f"\nInput shape: {h.shape}")

    # Forward pass
    probs, mask, stats = head(h)

    print(f"Halting probs shape: {probs.shape}")
    print(f"Mean halt prob: {stats['act_prob_mean']:.3f}")
    print(f"Halt rate: {stats['act_open_rate']:.3f}")
    print(f"Tokens halted: {mx.sum(mask).item()}/{mask.size}")

    print("\n✓ ACT Halting Head working!")


def demo_liquid_cell():
    """Test Liquid Time Constant Cell."""
    print("\n" + "=" * 80)
    print("Liquid Time Constant Cell Demo")
    print("=" * 80)

    cell = LiquidTimeConstantMLX(input_size=64, hidden_size=64)

    # Test input
    batch_size = 4
    x = mx.random.normal((batch_size, 64))
    h = mx.zeros((batch_size, 64))
    t = mx.array([1.0])

    print(f"\nInput shape: {x.shape}")
    print(f"Hidden shape: {h.shape}")
    print(f"Time: {t.item()}")

    # Forward pass
    h_new, output = cell(x, h, t)

    print(f"New hidden shape: {h_new.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden delta (L2): {mx.linalg.norm(h_new - h).item():.3f}")

    print("\n✓ Liquid Cell working!")


def demo_z5_scheduler():
    """Test Z5 Scheduler."""
    print("\n" + "=" * 80)
    print("Z5 Scheduler Demo")
    print("=" * 80)

    times = mx.arange(20).reshape(1, -1)
    slots = z5_slots(times)
    commits = boundary_commit_mask(times)

    print("\nTime sequence with Z5 slots:")
    print(f"{'Time':<8} {'Slot':<8} {'Commit?':<10}")
    print("─" * 26)

    for t in range(times.shape[1]):
        time_val = int(times[0, t].item())
        slot_val = int(slots[0, t].item())
        commit_val = "YES" if bool(commits[0, t].item()) else "no"
        marker = " ←" if commit_val == "YES" else ""
        print(f"{time_val:<8} {slot_val:<8} {commit_val:<10}{marker}")

    print("\n✓ Z5 Scheduler working!")


def demo_integration():
    """Test xLSTM block + Cube-Gated Block integration."""
    print("\n" + "=" * 80)
    print("xLSTM + HRM Integration Demo")
    print("=" * 80)

    from xlstm_metal.blocks.mlstm_mlx.xlstm_block import xLSTMBlock, xLSTMBlockConfig

    # Create xLSTM block
    xlstm_config = xLSTMBlockConfig(
        embedding_dim=256,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )
    xlstm_block = xLSTMBlock(xlstm_config)

    # Create HRM cube-gated block
    hrm_block = CubeGatedBlockMLX(
        d_in=256,
        fuse_phase_keys=True,
        max_items=50,
        topk=5
    )

    # Test input
    batch_size, seq_len = 2, 8
    x = mx.random.normal((batch_size, seq_len, 256))
    times = mx.broadcast_to(mx.arange(seq_len).reshape(1, -1), (batch_size, seq_len))

    print(f"\nInput shape: {x.shape}")

    # xLSTM forward
    print("\n1. xLSTM block forward...")
    xlstm_out, xlstm_state = xlstm_block(x)
    print(f"   Output shape: {xlstm_out.shape}")

    # HRM forward
    print("\n2. HRM cube-gated block forward...")
    hrm_state = {'times': times, 'train': False}
    hrm_out, hrm_telemetry = hrm_block(xlstm_out, hrm_state)
    print(f"   Output shape: {hrm_out.shape}")
    print(f"   Alpha: {hrm_telemetry['alpha_mean']:.3f}")
    print(f"   Confidence: {hrm_telemetry['conf_mean']:.3f}")

    print("\n✓ xLSTM + HRM integration working!")
    print("\nThis shows how HRM blocks can be inserted after xLSTM blocks")
    print("in a MAD wiring graph for memory-augmented predictions.")


if __name__ == "__main__":
    demo_memory_cube()
    demo_cube_gated_block()
    demo_act_halting()
    demo_liquid_cell()
    demo_z5_scheduler()
    demo_integration()

    print("\n" + "=" * 80)
    print("All HRM+ Blocks Working! ✓")
    print("=" * 80)
    print("\nThese blocks are now ready to be wired into MAD graphs.")
    print("Use create_hrm_xlstm_7b_wiring() to create full models.")
