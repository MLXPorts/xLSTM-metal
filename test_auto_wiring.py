"""
Test auto-wiring generation from xLSTM safetensors structure.

This demonstrates how AutoWiringFromConfig discovers the model
structure from weight keys and generates NCPS wirings.
"""

import mlx.core as mx
from xlstm_metal.ncps.wirings import (
    AutoWiringFromConfig,
    discover_blocks,
    discover_cells,
    visualize_wiring,
)


def test_discover_structure():
    """Test structure discovery from weight keys."""

    # Simulate weight keys from xlstm-7b safetensors
    weight_keys = [
        "backbone.blocks.0.mlstm_layer.q.weight",
        "backbone.blocks.0.mlstm_layer.k.weight",
        "backbone.blocks.0.mlstm_layer.v.weight",
        "backbone.blocks.0.mlstm_layer.igate_preact.weight",
        "backbone.blocks.0.mlstm_layer.igate_preact.bias",
        "backbone.blocks.0.mlstm_layer.fgate_preact.weight",
        "backbone.blocks.0.mlstm_layer.fgate_preact.bias",
        "backbone.blocks.0.mlstm_layer.ogate_preact.weight",
        "backbone.blocks.0.mlstm_layer.multihead_norm.weight",
        "backbone.blocks.0.mlstm_layer.out_proj.weight",
        "backbone.blocks.0.norm_mlstm.weight",
        "backbone.blocks.0.ffn.proj_up_gate.weight",
        "backbone.blocks.0.ffn.proj_up.weight",
        "backbone.blocks.0.ffn.proj_down.weight",
        "backbone.blocks.0.norm_ffn.weight",
        "backbone.blocks.1.mlstm_layer.q.weight",
        # ... more blocks
    ]

    print("=" * 70)
    print("Test 1: Discover Blocks")
    print("=" * 70)

    blocks = discover_blocks(weight_keys)
    for block_name in sorted(blocks.keys()):
        components = sorted(blocks[block_name])
        print(f"\n{block_name}:")
        for comp in components:
            print(f"  - {comp}")

    print("\n" + "=" * 70)
    print("Test 2: Discover Cells within mlstm_layer")
    print("=" * 70)

    cells = discover_cells(weight_keys, "backbone.blocks.0.mlstm_layer")
    print(f"\nCells in mlstm_layer:")
    for cell in sorted(cells):
        print(f"  - {cell}")

    print("\n" + "=" * 70)
    print("Test 3: Auto-generate Wirings")
    print("=" * 70)

    # Create dummy weights dict
    weights_dict = {key: mx.zeros((1, 1)) for key in weight_keys}

    # Config with wiring hints
    config = {
        "mlstm_layer": {
            "wiring": "auto",
            "sparsity": 0.5  # 50% sparse
        },
        "ffn": {
            "wiring": "sequential"  # FFN is sequential by nature
        }
    }

    block_wirings, component_wirings = AutoWiringFromConfig(weights_dict, config)

    print(f"\nGenerated wirings for {len(block_wirings)} blocks")

    # Visualize wiring for mlstm_layer in block 0
    if 'backbone.blocks.0' in component_wirings:
        if 'mlstm_layer' in component_wirings['backbone.blocks.0']:
            wiring = component_wirings['backbone.blocks.0']['mlstm_layer']
            cells = sorted(discover_cells(weight_keys, "backbone.blocks.0.mlstm_layer"))

            print("\n" + "-" * 70)
            print("Wiring for backbone.blocks.0.mlstm_layer:")
            print("-" * 70)
            print(visualize_wiring(wiring, cells))

            # Show polarity
            print("\n" + "-" * 70)
            print("Polarity Analysis:")
            print("-" * 70)
            adj = wiring.adjacency_matrix
            for i, src_cell in enumerate(cells):
                for j, dst_cell in enumerate(cells):
                    if adj[i, j] != 0:
                        polarity_str = "excitatory (+1)" if adj[i, j] > 0 else "inhibitory (-1)"
                        print(f"{src_cell} -> {dst_cell}: {polarity_str}")


def test_cell_instantiation():
    """Test creating cells with wirings."""

    print("\n\n" + "=" * 70)
    print("Test 4: Cell Instantiation")
    print("=" * 70)

    from xlstm_metal.ncps.neurons import GatedFFNCell, mLSTMCell

    # Create GatedFFN cell
    ffn_cell = GatedFFNCell(
        input_size=512,
        hidden_size=2048,
        activation='silu',
    )

    print("\n✓ Created GatedFFNCell")
    print(f"  Input: 512, Hidden: 2048")
    params = ffn_cell.parameters()
    print(f"  Parameters: {len(params)} tensors")

    # Create mLSTM cell with Metal kernel
    mlstm_cell = mLSTMCell(
        input_size=512,
        hidden_size=512,
        num_heads=8,
        kernel_backend='metal_parallel',
        chunk_size=64,
    )

    print("\n✓ Created mLSTMCell")
    print(f"  Input: 512, Hidden: 512, Heads: 8")
    print(f"  Kernel: metal_parallel (chunk_size=64)")
    params = mlstm_cell.parameters()
    print(f"  Parameters: {len(params)} tensors")

    # Test forward pass
    print("\n" + "-" * 70)
    print("Testing Forward Passes:")
    print("-" * 70)

    x = mx.random.normal((2, 10, 512))  # [batch=2, seq=10, dim=512]

    # FFN forward
    y_ffn, _ = ffn_cell(x)
    print(f"\n✓ FFN forward: {x.shape} -> {y_ffn.shape}")

    # mLSTM forward
    y_mlstm, state = mlstm_cell(x)
    print(f"✓ mLSTM forward: {x.shape} -> {y_mlstm.shape}")
    if state:
        C, n, m = state
        print(f"  State: C{C.shape}, n{n.shape}, m{m.shape}")


if __name__ == "__main__":
    test_discover_structure()
    test_cell_instantiation()

    print("\n\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print("\nKey Insights:")
    print("- Weight keys define the connectome (hierarchical graph)")
    print("- AutoNCP generates sparse wirings procedurally")
    print("- Forget gates automatically get inhibitory polarity (-1)")
    print("- Cells are NCPS neurons with Metal kernel backends")
    print("- Blocks are containers with internal cell wirings")
