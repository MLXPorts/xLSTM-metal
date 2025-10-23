#!/usr/bin/env python
"""
Compare MLX vs PyTorch implementations component by component.

Uses real weights from safetensors to test for numerical differences.
"""

import torch
import mlx.core as mx
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Import our MLX implementations
from xlstm_metal.blocks.mlstm_mlx.components import RMSNorm as MLXRMSNorm, soft_cap as mlx_soft_cap


def compare_tensors(mlx_tensor, torch_tensor, name, rtol=1e-4, atol=1e-5):
    """Compare MLX and PyTorch tensors."""
    # Convert to numpy for comparison
    mlx_np = np.array(mlx_tensor)
    torch_np = torch_tensor.detach().cpu().numpy()

    # Check shapes match
    if mlx_np.shape != torch_np.shape:
        print(f"❌ {name}: Shape mismatch! MLX: {mlx_np.shape}, PyTorch: {torch_np.shape}")
        return False

    # Compute differences
    abs_diff = np.abs(mlx_np - torch_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-10)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)

    passed = np.allclose(mlx_np, torch_np, rtol=rtol, atol=atol)

    status = "✅" if passed else "❌"
    print(f"{status} {name}:")
    print(f"  Max abs diff: {max_abs_diff:.2e}")
    print(f"  Max rel diff: {max_rel_diff:.2e}")
    print(f"  Mean abs diff: {mean_abs_diff:.2e}")

    if not passed:
        # Show where differences occur
        diff_mask = abs_diff > atol
        if diff_mask.sum() > 0:
            print(f"  Mismatches: {diff_mask.sum()}/{diff_mask.size} values ({100*diff_mask.sum()/diff_mask.size:.2f}%)")
            # Show a few examples
            indices = np.where(diff_mask)
            for i in range(min(5, len(indices[0]))):
                idx = tuple(ind[i] for ind in indices)
                print(f"    [{idx}]: MLX={mlx_np[idx]:.6f}, PyTorch={torch_np[idx]:.6f}, diff={abs_diff[idx]:.2e}")

    return passed


def test_soft_cap():
    """Test soft_cap function."""
    print("\n" + "="*80)
    print("Testing soft_cap function")
    print("="*80)

    # Test values
    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    cap_value = 15.0

    # MLX version
    x_mlx = mx.array(x_np)
    result_mlx = mlx_soft_cap(x_mlx, cap_value)

    # PyTorch version (from transformers)
    x_torch = torch.from_numpy(x_np)
    result_torch = cap_value * torch.tanh(x_torch / cap_value)

    compare_tensors(result_mlx, result_torch, "soft_cap", rtol=1e-5, atol=1e-6)


def test_rmsnorm():
    """Test RMSNorm implementation."""
    print("\n" + "="*80)
    print("Testing RMSNorm")
    print("="*80)

    # Create test input
    batch_size, seq_len, d_model = 2, 5, 128
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    weight_np = np.random.randn(d_model).astype(np.float32)

    # MLX version
    mlx_norm = MLXRMSNorm(
        num_features=d_model,
        eps=1e-6,
        use_weight=True,
        use_bias=False,
        force_float32_reductions=True
    )
    mlx_norm.weight = mx.array(weight_np)
    x_mlx = mx.array(x_np)
    result_mlx = mlx_norm(x_mlx)

    # PyTorch version (manual RMSNorm)
    x_torch = torch.from_numpy(x_np)
    weight_torch = torch.from_numpy(weight_np)

    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    variance = torch.mean(x_torch ** 2, dim=-1, keepdim=True)
    x_norm = x_torch * torch.rsqrt(variance + 1e-6)
    result_torch = weight_torch * x_norm

    compare_tensors(result_mlx, result_torch, "RMSNorm", rtol=1e-4, atol=1e-5)


def test_with_real_model():
    """Test using real model weights from transformers."""
    print("\n" + "="*80)
    print("Testing with real xLSTM-7B model")
    print("="*80)

    try:
        # Load transformers model
        print("Loading transformers xLSTM-7B model...")
        model_path = "xlstm_7b_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Try to load with LM head
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
            print("Loaded model with LM head (AutoModelForCausalLM)")
        except:
            model = AutoModel.from_pretrained(model_path, dtype=torch.float32)
            print("Loaded model without LM head (AutoModel) - will compare hidden states")

        model.eval()

        # Test input
        text = "The capital of France is"
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        print(f"\nInput text: '{text}'")
        print(f"Input IDs shape: {input_ids.shape}")

        # Get PyTorch output
        with torch.no_grad():
            torch_output = model(input_ids)

            # Check if we have logits or just hidden states
            if hasattr(torch_output, 'logits'):
                torch_logits = torch_output.logits
                print(f"PyTorch logits shape: {torch_logits.shape}")
                print(f"PyTorch logits stats: min={torch_logits.min():.4f}, max={torch_logits.max():.4f}, mean={torch_logits.mean():.4f}")
            else:
                torch_hidden = torch_output[0]
                print(f"PyTorch hidden states shape: {torch_hidden.shape}")
                print(f"PyTorch hidden stats: min={torch_hidden.min():.4f}, max={torch_hidden.max():.4f}, mean={torch_hidden.mean():.4f}")

        # Now compare with our MLX model
        from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner

        runner = xLSTM7BRunner()
        runner.load_weights(model_path)

        # Convert input to MLX
        input_ids_np = input_ids.numpy()
        input_ids_mlx = mx.array(input_ids_np)

        # Get MLX output (always returns logits)
        mlx_logits, _ = runner.forward(input_ids_mlx)

        print(f"\nMLX output shape: {mlx_logits.shape}")
        print(f"MLX logits stats: min={float(mx.min(mlx_logits)):.4f}, max={float(mx.max(mlx_logits)):.4f}, mean={float(mx.mean(mlx_logits)):.4f}")

        # Compare if we have matching outputs
        if hasattr(torch_output, 'logits'):
            print("\n✅ Both models produce logits - comparing...")
            compare_tensors(mlx_logits, torch_logits, "Full model logits", rtol=1e-2, atol=1e-2)
        else:
            print("\n⚠️  PyTorch model only produces hidden states, MLX produces logits")
            print("   Cannot directly compare - need to extract hidden states from MLX model")

    except Exception as e:
        print(f"Error testing with real model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*80)
    print("MLX vs PyTorch Component Comparison")
    print("="*80)

    # Test basic components
    test_soft_cap()
    test_rmsnorm()

    # Test with real model (if available)
    test_with_real_model()

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
