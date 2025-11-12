#!/usr/bin/env python3
"""Test script to verify norm_reduction_force_float32 override fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_block import mLSTMBlock
import mlx.core as mx


def test_norm_override():
    """Test that norm_reduction_force_float32 can be overridden."""

    # Mock config (sets norm_reduction_force_float32=False in config)
    config = {
        'embedding_dim': 512,
        'num_heads': 4,
        'qk_dim_factor': 0.5,
        'v_dim_factor': 1.0,
        'gate_soft_cap': 15.0,
        'norm_reduction_force_float32': False,  # Config says False
        'autocast_kernel_dtype': 'float32',
        'inference_state_dtype': 'float32',
        'chunk_size': 64,
        'norm_eps': 1e-6,
        'use_bias': False,
        'eps': 1e-6,
    }

    # Create block with override (should use True, not config's False)
    block = mLSTMBlock.from_config(
        block_index=0,
        config=config,
        norm_reduction_force_float32=True  # Override to True
    )

    # Verify the override was applied
    assert block.norm_reduction_force_float32 == True, \
        "norm_reduction_force_float32 override was not applied!"

    print("✓ Override test passed: norm_reduction_force_float32=True respected")

    # Test without override (should use config value)
    block2 = mLSTMBlock.from_config(
        block_index=0,
        config=config
    )

    assert block2.norm_reduction_force_float32 == False, \
        "Config default not working when no override provided"

    print("✓ Config test passed: Falls back to config value when no override")

    print("\n✅ All tests passed! The fix is working correctly.")


if __name__ == "__main__":
    test_norm_override()

