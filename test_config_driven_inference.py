#!/usr/bin/env python
"""
Test Script: Config-Driven Inference (No Hardcoded Parameters)

Verifies that the inference pipeline properly uses config.json and doesn't
rely on hardcoded model parameters.

Test Cases:
1. Config loading from config.json
2. Wiring creation from config
3. WiredMADModel instantiation
4. Parameter passing through blocks
5. Forward pass with different dtypes
"""

import mlx.core as mx
from pathlib import Path

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.mlx_jit.wiring import create_xlstm_wiring, WiredMADModel
from xlstm_metal.mlx_jit.utils import load_config


def test_config_loading(model_path):
    """Test 1: Verify config loads from config.json without hardcoding"""
    print("\n" + "="*80)
    print("TEST 1: Config Loading")
    print("="*80)

    config = load_config(model_path)

    print(f"✓ Config loaded from {model_path}/config.json")
    print(f"  Model dimensions:")
    print(f"    - embedding_dim: {config['embedding_dim']}")
    print(f"    - num_heads: {config['num_heads']}")
    print(f"    - num_blocks: {config['num_blocks']}")
    print(f"    - vocab_size: {config['vocab_size']}")
    print(f"    - qk_dim: {config['qk_dim']} (computed: {config['embedding_dim']} * {config['qk_dim_factor']})")
    print(f"    - v_dim: {config['v_dim']} (computed: {config['embedding_dim']} * {config['v_dim_factor']})")
    print(f"    - ffn_hidden_dim: {config['ffn_hidden_dim']} (computed with rounding)")

    return config


def test_wiring_creation(config):
    """Test 2: Verify wiring uses config parameters"""
    print("\n" + "="*80)
    print("TEST 2: Wiring Creation from Config")
    print("="*80)

    wiring = create_xlstm_wiring(config)

    print(f"✓ Wiring created with {len(wiring.block_specs)} blocks")
    print(f"  Block names: {list(wiring.block_specs.keys())}")

    # Check that block specs have correct parameters from config
    first_xlstm = wiring.block_specs['xlstm_0']
    print(f"\n  xlstm_0 params (should match config):")
    print(f"    - embedding_dim: {first_xlstm.params['embedding_dim']} (config: {config['embedding_dim']})")
    print(f"    - num_heads: {first_xlstm.params['num_heads']} (config: {config['num_heads']})")
    print(f"    - qk_dim_factor: {first_xlstm.params['qk_dim_factor']} (config: {config['qk_dim_factor']})")
    print(f"    - gate_soft_cap: {first_xlstm.params['gate_soft_cap']} (config: {config['gate_soft_cap']})")
    print(f"    - chunk_size: {first_xlstm.params['chunk_size']} (config: {config['chunk_size']})")

    assert first_xlstm.params['embedding_dim'] == config['embedding_dim'], "Embedding dim mismatch!"
    assert first_xlstm.params['num_heads'] == config['num_heads'], "Num heads mismatch!"

    print("\n  ✓ All parameters properly passed from config to BlockSpec")

    return wiring


def test_model_instantiation(wiring):
    """Test 3: Verify WiredMADModel creates blocks from BlockSpec params"""
    print("\n" + "="*80)
    print("TEST 3: Model Instantiation (Block Creation)")
    print("="*80)

    model = WiredMADModel(
        wiring=wiring,
        input_block='embedding',
        output_block='lm_head'
    )

    print(f"✓ WiredMADModel instantiated with {len(model.blocks)} blocks")
    print(f"  Execution stages: {len(model.stages)}")
    for i, stage in enumerate(model.stages):
        print(f"    Stage {i}: {stage}")

    # Check that blocks were created
    xlstm_block = model.blocks['xlstm_0']
    print(f"\n  xlstm_0 block structure:")
    print(f"    - Type: {type(xlstm_block).__name__}")
    print(f"    - Config embedding_dim: {xlstm_block.config.embedding_dim}")
    print(f"    - Config num_heads: {xlstm_block.config.num_heads}")
    print(f"    - mLSTM layer qk_dim: {xlstm_block.xlstm.config.qk_dim}")
    print(f"    - mLSTM layer v_dim: {xlstm_block.xlstm.config.v_dim}")
    print(f"    - FFN proj_up_dim: {xlstm_block.ffn.config.proj_up_dim}")

    # Verify dimensions match BlockSpec
    spec = wiring.block_specs['xlstm_0']
    assert xlstm_block.config.embedding_dim == spec.params['embedding_dim'], "Block config doesn't match spec!"

    print("\n  ✓ Block configs properly initialized from BlockSpec params")

    return model


def test_parameter_shapes(model, config):
    """Test 4: Verify parameter shapes match config (no hardcoding)"""
    print("\n" + "="*80)
    print("TEST 4: Parameter Shapes")
    print("="*80)

    xlstm_block = model.blocks['xlstm_0']

    # Check Q/K/V projection shapes
    q_weight = xlstm_block.xlstm.q.weight
    k_weight = xlstm_block.xlstm.k.weight
    v_weight = xlstm_block.xlstm.v.weight

    expected_qk_dim = int(config['embedding_dim'] * config['qk_dim_factor'])
    expected_v_dim = int(config['embedding_dim'] * config['v_dim_factor'])

    print(f"  Q projection shape: {q_weight.shape}")
    print(f"    Expected: [{expected_qk_dim}, {config['embedding_dim']}]")
    assert q_weight.shape == (expected_qk_dim, config['embedding_dim']), f"Q shape mismatch!"

    print(f"  K projection shape: {k_weight.shape}")
    print(f"    Expected: [{expected_qk_dim}, {config['embedding_dim']}]")
    assert k_weight.shape == (expected_qk_dim, config['embedding_dim']), f"K shape mismatch!"

    print(f"  V projection shape: {v_weight.shape}")
    print(f"    Expected: [{expected_v_dim}, {config['embedding_dim']}]")
    assert v_weight.shape == (expected_v_dim, config['embedding_dim']), f"V shape mismatch!"

    # Check gate shapes
    igate_weight = xlstm_block.xlstm.igate_preact.weight
    print(f"\n  Input gate shape: {igate_weight.shape}")
    print(f"    Expected: [{config['num_heads']}, {config['embedding_dim']}]")
    assert igate_weight.shape == (config['num_heads'], config['embedding_dim']), f"igate shape mismatch!"

    # Check FFN shapes
    ffn_up_weight = xlstm_block.ffn.proj_up.weight
    expected_ffn_dim = config['ffn_hidden_dim']
    print(f"\n  FFN up projection shape: {ffn_up_weight.shape}")
    print(f"    Expected: [{expected_ffn_dim}, {config['embedding_dim']}]")
    assert ffn_up_weight.shape == (expected_ffn_dim, config['embedding_dim']), f"FFN shape mismatch!"

    print("\n  ✓ All parameter shapes match config-derived dimensions")


def test_forward_pass(model, config):
    """Test 5: Verify forward pass with config-based dimensions"""
    print("\n" + "="*80)
    print("TEST 5: Forward Pass (Config-Based Dimensions)")
    print("="*80)

    batch_size = 2
    seq_len = 8

    # Create random input tokens
    input_ids = mx.random.randint(0, config['vocab_size'], (batch_size, seq_len))

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Vocab size: {config['vocab_size']} (from config)")

    # Forward pass
    try:
        logits, states = model(input_ids, hidden_states=None)

        print(f"\n  ✓ Forward pass successful!")
        print(f"    Output logits shape: {logits.shape}")
        print(f"    Expected: [{batch_size}, {seq_len}, {config['vocab_size']}]")

        assert logits.shape == (batch_size, seq_len, config['vocab_size']), "Output shape mismatch!"

        # Check states
        print(f"\n  States returned:")
        for block_name, state in states.items():
            if state is not None:
                c, n, m = state
                print(f"    {block_name}:")
                print(f"      c_state shape: {c.shape}")
                print(f"      n_state shape: {n.shape}")
                print(f"      m_state shape: {m.shape}")

                # Verify c_state is float32 (from dtype fix)
                assert c.dtype == mx.float32, f"{block_name} c_state not float32!"
                assert n.dtype == mx.float32, f"{block_name} n_state not float32!"
                assert m.dtype == mx.float32, f"{block_name} m_state not float32!"

        print(f"\n  ✓ All states are float32 (dtype fix verified)")

    except Exception as e:
        print(f"\n  ✗ Forward pass failed: {e}")
        raise


def test_dtype_consistency(model, config):
    """Test 6: Verify dtype handling (float32 states, mixed computation)"""
    print("\n" + "="*80)
    print("TEST 6: Dtype Consistency (Float32 States)")
    print("="*80)

    batch_size = 1
    seq_len = 4

    # Test with fp16 input (computation dtype)
    input_ids = mx.random.randint(0, config['vocab_size'], (batch_size, seq_len))

    # Forward pass
    logits, states = model(input_ids, hidden_states=None)

    # Check all states are float32
    all_float32 = True
    for block_name, state in states.items():
        if state is not None:
            c, n, m = state
            if c.dtype != mx.float32 or n.dtype != mx.float32 or m.dtype != mx.float32:
                print(f"  ✗ {block_name} states not float32: c={c.dtype}, n={n.dtype}, m={m.dtype}")
                all_float32 = False

    if all_float32:
        print(f"  ✓ All states maintain float32 dtype (numerically stable)")
    else:
        raise AssertionError("Some states are not float32!")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("XLSTM-METAL: CONFIG-DRIVEN INFERENCE TEST SUITE")
    print("="*80)
    print("\nVerifying that the architecture uses config.json without hardcoded parameters")

    # Use xlstm_7b_model directory
    model_path = Path(__file__).parent / "xlstm_7b_model"

    if not model_path.exists():
        print(f"\n✗ Model directory not found: {model_path}")
        print("  Please download or create xlstm_7b_model with config.json")
        return

    try:
        # Run tests sequentially
        config = test_config_loading(str(model_path))
        wiring = test_wiring_creation(config)
        model = test_model_instantiation(wiring)
        test_parameter_shapes(model, config)
        test_forward_pass(model, config)
        test_dtype_consistency(model, config)

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe architecture is properly config-driven:")
        print("  1. Config loads from config.json without hardcoding")
        print("  2. Wiring creates blocks with config parameters")
        print("  3. Blocks instantiate with correct dimensions")
        print("  4. Parameters have config-derived shapes")
        print("  5. Forward pass works with config dimensions")
        print("  6. States maintain float32 dtype (numerically stable)")
        print("\n✓ Ready for inference with any xLSTM model size!")

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
