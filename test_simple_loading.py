#!/usr/bin/env python3
"""
Simple test to verify model loading pipeline works step by step.

This tests the path: safetensors -> config -> model creation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from xlstm_metal.mlx_jit.load_model import load_config


def test_config_loading():
    """Test 1: Can we load config.json?"""
    print("=" * 60)
    print("TEST 1: Loading config.json")
    print("=" * 60)
    
    model_dir = Path("xlstm_7b_model")
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
        
    try:
        config = load_config(str(model_dir))
        print(f"✓ Config loaded successfully")
        print(f"  - embedding_dim: {config['embedding_dim']}")
        print(f"  - num_heads: {config['num_heads']}")
        print(f"  - num_blocks: {config['num_blocks']}")
        print(f"  - vocab_size: {config['vocab_size']}")
        print(f"  - chunk_size: {config['chunk_size']}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safetensors_loading():
    """Test 2: Can we load safetensors weights?"""
    print("\n" + "=" * 60)
    print("TEST 2: Loading safetensors weights")
    print("=" * 60)
    
    model_dir = Path("xlstm_7b_model")
    index_file = model_dir / "model.safetensors.index.json"
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return False
        
    try:
        import json
        with open(index_file) as f:
            index = json.load(f)
            
        print(f"✓ Index loaded: {len(index['weight_map'])} weight tensors")
        
        # Load first shard as test
        shard_files = sorted(set(index['weight_map'].values()))
        first_shard = model_dir / shard_files[0]
        
        print(f"  Loading test shard: {first_shard.name}")
        weights = mx.load(str(first_shard))
        print(f"✓ Loaded {len(weights)} tensors from first shard")
        
        # Show a few keys
        keys = list(weights.keys())[:5]
        for key in keys:
            print(f"    - {key}: {weights[key].shape}")
            
        return True
    except Exception as e:
        print(f"❌ Safetensors loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_cell_creation():
    """Test 3: Can we create a single xLSTM block cell?"""
    print("\n" + "=" * 60)
    print("TEST 3: Creating xLSTM block cell")
    print("=" * 60)
    
    try:
        config = load_config("xlstm_7b_model")
        
        # Import the cell
        from xlstm_metal.mlx_jit.models.xlstm_7b_model import xLSTM7BCell
        
        # Create a single block (block 0)
        cell = xLSTM7BCell.from_config(block_index=0, config=config)
        
        print(f"✓ xLSTM block cell created successfully")
        print(f"  - block_index: {cell.block_index}")
        print(f"  - embedding_dim: {cell.embedding_dim}")
        print(f"  - num_heads: {cell.num_heads}")
        print(f"  - qk_dim_per_head: {cell.qk_dim_per_head}")
        print(f"  - v_dim_per_head: {cell.v_dim_per_head}")
        
        # Check parameters exist
        print(f"\n  Parameters:")
        print(f"    - norm_mlstm: {cell.norm_mlstm.weight.shape}")
        print(f"    - mlstm_cell.q_proj: {cell.mlstm_cell.q_proj.weight.shape}")
        print(f"    - mlstm_cell.k_proj: {cell.mlstm_cell.k_proj.weight.shape}")
        print(f"    - mlstm_cell.v_proj: {cell.mlstm_cell.v_proj.weight.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Cell creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test 4: Can we run a forward pass through a cell?"""
    print("\n" + "=" * 60)
    print("TEST 4: Running forward pass")
    print("=" * 60)
    
    try:
        config = load_config("xlstm_7b_model")
        from xlstm_metal.mlx_jit.models.xlstm_7b_model import xLSTM7BCell
        
        cell = xLSTM7BCell.from_config(block_index=0, config=config)
        
        # Create dummy input: [batch=1, seq_len=4, embedding_dim]
        B, S, D = 1, 4, config['embedding_dim']
        x = mx.random.normal(shape=(B, S, D), dtype=mx.float32)
        
        print(f"  Input shape: {x.shape}")
        
        # Forward pass
        output, state = cell(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  State shapes:")
        print(f"    - C: {state[0].shape}")
        print(f"    - n: {state[1].shape}")  
        print(f"    - m: {state[2].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\nTesting xLSTM-Metal Model Loading Pipeline")
    print("=" * 60)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Safetensors Loading", test_safetensors_loading),
        ("Cell Creation", test_model_cell_creation),
        ("Forward Pass", test_forward_pass),
    ]
    
    results = []
    for name, test_fn in tests:
        success = test_fn()
        results.append((name, success))
        
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        
    all_passed = all(success for _, success in results)
    print("\n" + ("✓ All tests passed!" if all_passed else "❌ Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
