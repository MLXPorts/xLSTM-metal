#!/usr/bin/env python
"""
HRM+ xLSTM-7B Demo

Demonstrates the integration of HRM+ (Hierarchical Retrieval Memory) components
with the xLSTM-7B model using the MAD (Modular Atomically-wired Differentiable) framework.

This example shows:
1. Creating HRM-enhanced xLSTM-7B wiring with different strategies
2. Running forward passes with temporal information (Z5 scheduler)
3. Collecting HRM telemetry (alpha gates, cube confidence, ACT halting)
4. Optional neuromodulation (5-HT serotonin)
"""

import mlx.core as mx
from xlstm_metal.wiring.mlx import create_hrm_xlstm_7b_wiring, WiredMADModel


def demo_hrm_strategies():
    """Compare different HRM integration strategies."""

    print("=" * 80)
    print("HRM+ xLSTM-7B Integration Demo")
    print("=" * 80)

    # Small model for demo (2 blocks instead of 32)
    embedding_dim = 512
    num_blocks = 2
    num_heads = 8
    vocab_size = 1000
    batch_size = 2
    seq_len = 16

    strategies = [
        ("none", "Standard xLSTM without HRM"),
        ("per_segment", "HRM gates every 2 blocks"),
        ("post_process", "Single HRM wrapper at end"),
    ]

    for strategy, description in strategies:
        print(f"\n{'─' * 80}")
        print(f"Strategy: {strategy}")
        print(f"Description: {description}")
        print(f"{'─' * 80}\n")

        # Create wiring
        wiring = create_hrm_xlstm_7b_wiring(
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            vocab_size=vocab_size,
            hrm_strategy=strategy,
            hrm_segment_size=2,
            enable_act=True,
            fuse_phase_keys=True,
            cube_max_items=1024,  # Small for demo
            cube_topk=8
        )

        # Instantiate model
        model = WiredMADModel(wiring, 'embedding', 'lm_head')

        # Generate dummy input
        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        # Generate time steps for Z5 scheduler
        times = mx.arange(seq_len).reshape(1, -1)
        times = mx.broadcast_to(times, (batch_size, seq_len))

        print(f"Input shape: {input_ids.shape}")
        print(f"Times shape: {times.shape}")
        print(f"Times (Z5 slots): {times[0].tolist()}")

        # Forward pass
        print("\nRunning forward pass...")
        output, hidden_states = model(input_ids)

        print(f"Output shape: {output.shape}")
        print(f"Hidden states keys: {list(hidden_states.keys())}")

        # Check for HRM telemetry (only available if HRM blocks present)
        if strategy != "none":
            print("\n✓ HRM+ integration active")
            # Note: Telemetry collection requires modifications to forward pass
            # to thread through times and collect stats
        else:
            print("\n✓ Standard xLSTM (no HRM)")


def demo_hrm_telemetry():
    """Demonstrate HRM telemetry collection."""

    print("\n\n" + "=" * 80)
    print("HRM+ Telemetry Demo")
    print("=" * 80)

    from xlstm_metal.blocks.hrm_mlx import HRMxLSTMBlockMLX, HRMxLSTMConfig
    from xlstm_metal.blocks.mlstm_mlx.xlstm_block import xLSTMBlockConfig

    # Create a single HRM-enhanced block
    xlstm_config = xLSTMBlockConfig(
        embedding_dim=512,
        num_heads=8,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )

    hrm_config = HRMxLSTMConfig(
        xlstm_config=xlstm_config,
        enable_hrm=True,
        enable_act=True,
        fuse_phase_keys=True
    )

    block = HRMxLSTMBlockMLX(hrm_config)

    # Generate input
    batch_size, seq_len, d_model = 2, 16, 512
    x = mx.random.normal((batch_size, seq_len, d_model))
    times = mx.arange(seq_len).reshape(1, -1).broadcast_to((batch_size, seq_len))

    print(f"\nInput shape: {x.shape}")
    print(f"Times shape: {times.shape}")

    # Forward pass
    print("\nRunning HRM-enhanced forward pass...")
    output, state, telemetry = block(x, times=times, train=False)

    print(f"\nOutput shape: {output.shape}")
    print(f"\nTelemetry:")
    for key, value in telemetry.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {type(value).__name__} {getattr(value, 'shape', '')}")


def demo_neuromodulation():
    """Demonstrate neuromodulation (5-HT serotonin)."""

    print("\n\n" + "=" * 80)
    print("Neuromodulation Demo (5-HT Serotonin)")
    print("=" * 80)

    from xlstm_metal.blocks.hrm_mlx import CubeGatedBlockMLX

    # Create cube-gated block
    block = CubeGatedBlockMLX(
        d_in=512,
        fuse_phase_keys=True,
        k_5ht=0.5,  # Modulation strength
        gain_floor=0.3  # Minimum gain
    )

    batch_size, seq_len, d_model = 2, 16, 512
    h_in = mx.random.normal((batch_size, seq_len, d_model))
    times = mx.arange(seq_len).reshape(1, -1).broadcast_to((batch_size, seq_len))

    # Test different serotonin levels
    mod_levels = [0.0, 0.5, 1.0]

    print("\nComparing different 5-HT levels:")
    print(f"{'5-HT Level':<15} {'Alpha Mean':<15} {'Conf Mean':<15} {'Energy Ratio':<15}")
    print("─" * 60)

    for level in mod_levels:
        mod_5ht = mx.ones((batch_size, seq_len)) * level

        # Forward pass
        y_out, alpha_mean, conf_mean = block(
            h_in=h_in,
            times=times,
            mod_5ht=mod_5ht,
            train=False
        )

        energy_in = float(mx.mean(mx.sum(h_in ** 2, axis=-1)).item())
        energy_out = float(mx.mean(mx.sum(y_out ** 2, axis=-1)).item())
        energy_ratio = energy_out / energy_in if energy_in > 0 else 0

        print(f"{level:<15.2f} {alpha_mean:<15.4f} {conf_mean:<15.4f} {energy_ratio:<15.4f}")

    print("\nNote: Higher 5-HT → lower gain → reduced memory influence")


def demo_z5_scheduler():
    """Demonstrate Z5 temporal discretization."""

    print("\n\n" + "=" * 80)
    print("Z5 Scheduler Demo (Temporal Discretization)")
    print("=" * 80)

    from xlstm_metal.blocks.hrm_mlx import z5_slots, boundary_commit_mask

    # Generate time sequence
    times = mx.arange(20).reshape(1, -1)

    # Compute Z5 slots and boundary commits
    slots = z5_slots(times)
    commits = boundary_commit_mask(times)

    print("\nTime sequence with Z5 slots and boundary commits:")
    print(f"{'Time':<8} {'Slot':<8} {'Commit?':<10}")
    print("─" * 26)

    for t in range(times.shape[1]):
        time_val = int(times[0, t].item())
        slot_val = int(slots[0, t].item())
        commit_val = "YES" if bool(commits[0, t].item()) else "no"

        marker = " ←" if commit_val == "YES" else ""
        print(f"{time_val:<8} {slot_val:<8} {commit_val:<10}{marker}")

    print("\nNote: Boundary commits occur every 5th step (slot == 4)")
    print("      This enforces base-5 carry structure for temporal discretization")


if __name__ == "__main__":
    # Run all demos
    demo_hrm_strategies()
    demo_hrm_telemetry()
    demo_neuromodulation()
    demo_z5_scheduler()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Train HRM-enhanced model with ponder loss for ACT")
    print("2. Experiment with different hrm_strategy and hrm_segment_size")
    print("3. Tune k_5ht and gain_floor for neuromodulation")
    print("4. Analyze cube retrieval patterns and confidence scores")
    print("5. Scale up to full xLSTM-7B (32 blocks, 4096 dims)")
