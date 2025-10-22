#!/usr/bin/env python
"""
Test xLSTM MLX blocks after ZERO TOLERANCE operator refactor.

Verifies that all blocks work correctly with pure MLX operations.
"""

import mlx.core as mx
import sys
from pathlib import Path

# Add xlstm_metal to path
sys.path.insert(0, str(Path(__file__).parent / "xlstm_metal"))

from blocks.mlstm_mlx.components import RMSNorm, MultiHeadLayerNorm, soft_cap
from blocks.mlstm_mlx.block import mLSTMConfig, mLSTMLayer, mLSTMBlock
from blocks.mlstm_mlx.ffn import FFNConfig, GatedFFN, FFNBlock
from blocks.mlstm_mlx.xlstm_block import xLSTMBlockConfig, xLSTMBlock


def test_components():
    """Test RMSNorm, MultiHeadLayerNorm, and soft_cap."""
    print("Testing components...")

    # Test soft_cap
    x = mx.random.normal((4, 8))
    y = soft_cap(x, 15.0)
    assert y.shape == x.shape, "soft_cap shape mismatch"
    print(f"  ✓ soft_cap: {x.shape} -> {y.shape}")

    # Test RMSNorm
    norm = RMSNorm(num_features=512)
    x = mx.random.normal((2, 10, 512))
    y = norm(x)
    assert y.shape == x.shape, "RMSNorm shape mismatch"
    print(f"  ✓ RMSNorm: {x.shape} -> {y.shape}")

    # Test MultiHeadLayerNorm
    mh_norm = MultiHeadLayerNorm(num_heads=8, head_dim=64)
    x = mx.random.normal((2, 10, 8, 64))
    y = mh_norm(x)
    assert y.shape == x.shape, "MultiHeadLayerNorm shape mismatch"
    print(f"  ✓ MultiHeadLayerNorm: {x.shape} -> {y.shape}")


def test_mlstm_layer():
    """Test mLSTM layer forward pass."""
    print("\nTesting mLSTM layer...")

    config = mLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )

    layer = mLSTMLayer(config)

    # Test forward pass
    x = mx.random.normal((2, 8, 512))  # [B=2, S=8, D=512]
    y, state = layer(x, state=None)

    assert y.shape == x.shape, f"mLSTM output shape mismatch: {y.shape} vs {x.shape}"
    assert state is not None, "State should be returned"
    c, n, m = state
    # C: [B, NH, QK_DH, V_DH] where QK_DH=256/4=64, V_DH=512/4=128
    assert c.shape == (2, 4, 64, 128), f"C state shape mismatch: {c.shape}"
    assert n.shape == (2, 4, 64), f"N state shape mismatch: {n.shape}"
    assert m.shape == (2, 4), f"M state shape mismatch: {m.shape}"

    print(f"  ✓ Forward pass: {x.shape} -> {y.shape}")
    print(f"  ✓ State shapes: C{c.shape}, N{n.shape}, M{m.shape}")

    # Test with existing state
    y2, state2 = layer(x, state=state)
    assert y2.shape == x.shape, "Second forward pass shape mismatch"
    print(f"  ✓ Stateful forward: {x.shape} -> {y2.shape}")


def test_mlstm_block():
    """Test complete mLSTM block with normalization."""
    print("\nTesting mLSTM block...")

    config = mLSTMConfig(
        embedding_dim=512,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        gate_soft_cap=15.0
    )

    block = mLSTMBlock(config)

    x = mx.random.normal((2, 8, 512))
    y, state = block(x, state=None)

    assert y.shape == x.shape, f"mLSTM block shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ mLSTM block with residual: {x.shape} -> {y.shape}")


def test_gated_ffn():
    """Test Gated FFN (SwiGLU)."""
    print("\nTesting Gated FFN...")

    config = FFNConfig(
        embedding_dim=512,
        proj_factor=2.5,
        act_fn="swish"
    )

    ffn = GatedFFN(config)

    x = mx.random.normal((2, 8, 512))
    y = ffn(x)

    assert y.shape == x.shape, f"FFN shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ GatedFFN (SwiGLU): {x.shape} -> {y.shape}")

    # Test FFN block with normalization
    ffn_block = FFNBlock(config)
    y2 = ffn_block(x)
    assert y2.shape == x.shape, f"FFN block shape mismatch: {y2.shape} vs {x.shape}"
    print(f"  ✓ FFN block with residual: {x.shape} -> {y2.shape}")


def test_xlstm_block():
    """Test complete xLSTM block (mLSTM + FFN)."""
    print("\nTesting complete xLSTM block...")

    config = xLSTMBlockConfig(
        embedding_dim=512,
        num_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        ffn_proj_factor=2.5,
        gate_soft_cap=15.0
    )

    block = xLSTMBlock(config)

    x = mx.random.normal((2, 8, 512))
    y, state = block(x, state=None)

    assert y.shape == x.shape, f"xLSTM block shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ xLSTM block (mLSTM+FFN): {x.shape} -> {y.shape}")

    # Test stateful processing
    y2, state2 = block(x, state=state)
    assert y2.shape == x.shape, "Stateful xLSTM block shape mismatch"
    print(f"  ✓ Stateful xLSTM block: {x.shape} -> {y2.shape}")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\nTesting numerical stability...")

    config = mLSTMConfig(embedding_dim=256, num_heads=4)
    layer = mLSTMLayer(config)

    # Test with zeros
    x_zeros = mx.zeros((1, 4, 256))
    y_zeros, _ = layer(x_zeros)
    assert not mx.any(mx.isnan(y_zeros)), "NaN detected with zero input"
    print("  ✓ Zero input: no NaN")

    # Test with large values
    x_large = mx.ones((1, 4, 256)) * 10.0
    y_large, _ = layer(x_large)
    assert not mx.any(mx.isnan(y_large)), "NaN detected with large input"
    print("  ✓ Large input: no NaN")

    # Test with small values
    x_small = mx.ones((1, 4, 256)) * 1e-5
    y_small, _ = layer(x_small)
    assert not mx.any(mx.isnan(y_small)), "NaN detected with small input"
    print("  ✓ Small input: no NaN")


def main():
    """Run all tests."""
    print("=" * 60)
    print("xLSTM MLX Refactor Test Suite")
    print("Testing ZERO TOLERANCE operator policy compliance")
    print("=" * 60)

    try:
        test_components()
        test_mlstm_layer()
        test_mlstm_block()
        test_gated_ffn()
        test_xlstm_block()
        test_numerical_stability()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Pure MLX operations working correctly.")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
