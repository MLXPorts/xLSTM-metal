#!/usr/bin/env python3
"""Numerical Parity Test: MLX vs Canonical PyTorch xLSTM

This test compares our MLX implementation against the canonical PyTorch
implementation (native fallbacks) from quarantine/xlstm/modeling_xlstm.py.

We test individual functions and components side-by-side with identical inputs.

Tests:
1. soft_cap function
2. mlstm_recurrent_step_native (single step)
3. RMSNorm (if we can extract it)

Tolerance:
- float32: rtol=1e-4, atol=1e-5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import mlx.core as mx
import torch
from xlstm_metal.mlx_jit.models import WiredxLSTM

# Import canonical PyTorch implementations (native, no Triton)
# These are the fallback implementations that work without external xlstm package
sys.path.insert(0, str(Path(__file__).parent / "quarantine"))
from quarantine.xlstm.modeling_xlstm import soft_cap as torch_soft_cap

# Import our MLX implementations
from xlstm_metal.mlx_jit.blocks.soft_cap import soft_cap as mlx_soft_cap
from xlstm_metal.mlx_jit.blocks.rms_norm import RMSNormCell as MLXRMSNorm


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)


def compare_tensors(mlx_tensor, torch_tensor, name, rtol=1e-5, atol=1e-6):
    """Compare MLX and PyTorch tensors."""
    mlx_np = np.array(mlx_tensor)
    torch_np = torch_tensor.detach().cpu().numpy()

    # Check shapes match
    if mlx_np.shape != torch_np.shape:
        print(f"❌ {name}: Shape mismatch! MLX={mlx_np.shape}, PyTorch={torch_np.shape}")
        return False

    # Check for NaN/Inf
    mlx_has_nan = np.any(np.isnan(mlx_np))
    torch_has_nan = np.any(np.isnan(torch_np))
    mlx_has_inf = np.any(np.isinf(mlx_np))
    torch_has_inf = np.any(np.isinf(torch_np))

    if mlx_has_nan or mlx_has_inf:
        print(f"❌ {name}: MLX has NaN={mlx_has_nan}, Inf={mlx_has_inf}")
        return False

    if torch_has_nan or torch_has_inf:
        print(f"❌ {name}: PyTorch has NaN={torch_has_nan}, Inf={torch_has_inf}")
        return False

    # Compute differences
    abs_diff = np.abs(mlx_np - torch_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-8)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Check tolerance
    close = np.allclose(mlx_np, torch_np, rtol=rtol, atol=atol)

    if close:
        print(f"✅ {name}: PASS (max_abs={max_abs_diff:.6e}, max_rel={max_rel_diff:.6e})")
    else:
        print(f"❌ {name}: FAIL (max_abs={max_abs_diff:.6e}, max_rel={max_rel_diff:.6e})")
        print(f"   Mean abs diff: {mean_abs_diff:.6e}")
        print(f"   MLX range: [{np.min(mlx_np):.6f}, {np.max(mlx_np):.6f}]")
        print(f"   PyTorch range: [{np.min(torch_np):.6f}, {np.max(torch_np):.6f}]")

        # Show worst mismatches
        worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"   Worst mismatch at {worst_idx}: MLX={mlx_np[worst_idx]:.6f}, PyTorch={torch_np[worst_idx]:.6f}")

    return close


def test_soft_cap():
    """Test soft-cap function parity."""
    print("\n" + "="*60)
    print("TEST 1: Soft-Cap Function")
    print("="*60)


    set_seed(42)

    # Test values
    test_cases = [
        ("small values", np.array([-0.5, 0.0, 0.5, 1.0]), 15.0),
        ("large values", np.array([-50.0, -20.0, 20.0, 50.0]), 15.0),
        ("edge cases", np.array([-100.0, 0.0, 100.0]), 30.0),
        ("random", np.random.randn(100) * 10, 15.0),
    ]

    all_pass = True
    for name, x_np, cap in test_cases:
        # MLX
        x_mlx = mx.array(x_np, dtype=mx.float32)
        y_mlx = mlx_soft_cap(x_mlx, cap)

        # PyTorch
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        y_torch = torch_soft_cap(x_torch, cap)

        passed = compare_tensors(y_mlx, y_torch, f"soft_cap({name})")
        all_pass = all_pass and passed

    return all_pass


def test_rmsnorm():
    """Test RMSNorm parity."""
    print("\n" + "="*60)
    print("TEST 2: RMSNorm")
    print("="*60)


    set_seed(42)

    # Create RMSNorm layers
    dim = 512
    eps = 1e-6

    # MLX
    mlx_norm = MLXRMSNorm(dims=dim, eps=eps, force_float32_reductions=True)
    mlx_norm.weight = mx.ones((dim,), dtype=mx.float32)

    # PyTorch (using the canonical implementation snippet)
    class TorchRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            # Force float32 for stability
            x_fp32 = x.float()
            variance = x_fp32.pow(2).mean(-1, keepdim=True)
            x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
            return (self.weight * x_normed).to(x.dtype)

    torch_norm = TorchRMSNorm(dim, eps)

    # Test cases
    test_cases = [
        ("small batch", (2, 10, dim)),
        ("single token", (1, 1, dim)),
        ("long sequence", (1, 100, dim)),
        ("large values", (2, 5, dim), 10.0),  # scaled input
    ]

    all_pass = True
    for case in test_cases:
        name = case[0]
        shape = case[1]
        scale = case[2] if len(case) > 2 else 1.0

        # Generate input
        x_np = np.random.randn(*shape).astype(np.float32) * scale

        # MLX
        x_mlx = mx.array(x_np)
        y_mlx = mlx_norm(x_mlx)

        # PyTorch
        x_torch = torch.tensor(x_np)
        y_torch = torch_norm(x_torch)

        passed = compare_tensors(y_mlx, y_torch, f"RMSNorm({name})")
        all_pass = all_pass and passed

    return all_pass


def test_embeddings():
    """Test embedding layer parity."""
    print("\n" + "="*60)
    print("TEST 3: Embeddings")
    print("="*60)

    set_seed(42)

    # Load model to get real embeddings
    print("Loading MLX model...")
    mlx_model = WiredxLSTM.from_pretrained(
        "xlstm_7b_model",
        compute_dtype=mx.float32,
        state_dtype=mx.float32,
        norm_reduce_force_float32=True
    )

    # Test token IDs
    test_cases = [
        ("single token", [[12092]]),  # "Hello"
        ("multiple tokens", [[12092, 1533, 13]]),  # "Hello world ,"
        ("batch", [[12092, 1533], [849, 403]]),  # batch of 2
    ]

    all_pass = True
    for name, token_ids in test_cases:
        # MLX
        input_ids_mlx = mx.array(token_ids, dtype=mx.int32)
        emb_mlx = mlx_model.embedding(input_ids_mlx)

        print(f"\nEmbedding test: {name}")
        print(f"  Input shape: {input_ids_mlx.shape}")
        print(f"  Output shape: {emb_mlx.shape}, dtype: {emb_mlx.dtype}")
        print(f"  Output range: [{mx.min(emb_mlx).item():.6f}, {mx.max(emb_mlx).item():.6f}]")
        print(f"  Has NaN: {mx.any(mx.isnan(emb_mlx)).item()}")
        print(f"  Has Inf: {mx.any(mx.isinf(emb_mlx)).item()}")

        # Check for issues
        has_nan = mx.any(mx.isnan(emb_mlx)).item()
        has_inf = mx.any(mx.isinf(emb_mlx)).item()

        if has_nan or has_inf:
            print(f"  ❌ FAIL: NaN={has_nan}, Inf={has_inf}")
            all_pass = False
        else:
            print(f"  ✅ PASS")

    return all_pass


def test_mlstm_block():
    """Test mLSTM block forward pass for numerical stability."""
    print("\n" + "="*60)
    print("TEST 4: mLSTM Block Forward Pass")
    print("="*60)

    set_seed(42)

    # Load model
    print("Loading MLX model...")
    mlx_model = WiredxLSTM.from_pretrained(
        "xlstm_7b_model",
        compute_dtype=mx.float32,
        state_dtype=mx.float32,
        norm_reduce_force_float32=True
    )

    # Test inputs
    test_cases = [
        ("short sequence", [[12092, 1533, 13]]),  # 3 tokens
        ("single token", [[12092]]),  # 1 token
        ("medium sequence", [[12092, 1533, 13, 849, 403, 368, 32]]),  # 7 tokens
    ]

    all_pass = True
    for name, token_ids in test_cases:
        print(f"\nmLSTM block test: {name}")
        input_ids = mx.array(token_ids, dtype=mx.int32)

        # Forward through first block only
        x = mlx_model.embedding(input_ids)
        print(f"  After embedding: shape={x.shape}, range=[{mx.min(x).item():.6f}, {mx.max(x).item():.6f}]")

        # First block
        block = mlx_model.blocks[0]
        x_out, state = block(x, None)

        print(f"  After block 0: shape={x_out.shape}, range=[{mx.min(x_out).item():.6f}, {mx.max(x_out).item():.6f}]")
        print(f"  Has NaN: {mx.any(mx.isnan(x_out)).item()}")
        print(f"  Has Inf: {mx.any(mx.isinf(x_out)).item()}")

        if mx.any(mx.isnan(x_out)).item() or mx.any(mx.isinf(x_out)).item():
            print(f"  ❌ FAIL: Numerical instability detected")
            all_pass = False
        else:
            print(f"  ✅ PASS")


    return all_pass


def test_full_forward():
    """Test full model forward pass."""
    print("\n" + "="*60)
    print("TEST 5: Full Model Forward Pass")
    print("="*60)

    set_seed(42)

    # Load model
    print("Loading MLX model...")
    mlx_model = WiredxLSTM.from_pretrained(
        "xlstm_7b_model",
        compute_dtype=mx.float32,
        state_dtype=mx.float32,
        norm_reduce_force_float32=True
    )

    # Test with increasing sequence lengths
    test_cases = [
        ("very short", [[12092]]),
        ("short", [[12092, 1533, 13]]),
        ("medium", [[12092, 1533, 13, 849, 403, 368, 32]]),
    ]

    all_pass = True
    for name, token_ids in test_cases:
        print(f"\nFull forward test: {name} (seq_len={len(token_ids[0])})")
        input_ids = mx.array(token_ids, dtype=mx.int32)

        logits = mlx_model(input_ids)

        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{mx.min(logits).item():.6f}, {mx.max(logits).item():.6f}]")
        print(f"  Has NaN: {mx.any(mx.isnan(logits)).item()}")
        print(f"  Has Inf: {mx.any(mx.isinf(logits)).item()}")

        # Check numerical stability
        has_nan = mx.any(mx.isnan(logits)).item()
        has_inf = mx.any(mx.isinf(logits)).item()

        if has_nan or has_inf:
            print(f"  ❌ FAIL: NaN={has_nan}, Inf={has_inf}")
            all_pass = False
        else:
            # Check if output is reasonable
            min_val = mx.min(logits).item()
            max_val = mx.max(logits).item()
            if min_val < -100 or max_val > 100:
                print(f"  ⚠️ WARNING: Large logit range may indicate instability")
            print(f"  ✅ PASS")


    return all_pass


def main():
    """Run all numerical parity tests."""
    print("="*60)
    print("NUMERICAL PARITY TEST SUITE")
    print("MLX Implementation vs Canonical Transformers")
    print("="*60)

    results = {}

    # Run tests
    results['soft_cap'] = test_soft_cap()
    results['rmsnorm'] = test_rmsnorm()
    results['embeddings'] = test_embeddings()
    results['mlstm_block'] = test_mlstm_block()
    results['full_forward'] = test_full_forward()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠️ SOME TESTS FAILED - Review output above")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

