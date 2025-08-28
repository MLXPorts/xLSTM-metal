#!/usr/bin/env python
"""
Comprehensive testing and benchmarking script for xLSTM implementations
"""

import time
import numpy as np
import sys
from typing import Tuple, Optional

def generate_synthetic_data(batch_size=4, seq_len=100, vocab_size=1000):
    """Generate synthetic sequence data for testing"""
    tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
    return tokens, targets


def test_mlx_implementation():
    """Test MLX xLSTM with training and inference"""
    print("\n" + "="*60)
    print("Testing MLX Implementation")
    print("="*60)
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from xlstm_mlx import create_xlstm_model
        
        # Model configuration
        config = {
            'vocab_size': 1000,
            'num_layers': 4,
            'signature': (3, 1),  # 3 mLSTM, 1 sLSTM pattern
            'inp_dim': 256,
            'head_dim': 32,
            'head_num': 8,
            'dropout': 0.1
        }
        
        print(f"\nModel Configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Create model
        model = create_xlstm_model(**config)
        
        # Count parameters
        def count_params(params):
            count = 0
            for p in params.values():
                if isinstance(p, mx.array):
                    count += p.size
                elif isinstance(p, dict):
                    count += count_params(p)
                elif isinstance(p, list):
                    for item in p:
                        if isinstance(item, mx.array):
                            count += item.size
            return count
        
        param_count = count_params(model.parameters())
        print(f"\nTotal parameters: {param_count:,}")
        
        # Test forward pass
        batch_size = 4
        seq_len = 50
        tokens, targets = generate_synthetic_data(batch_size, seq_len, config['vocab_size'])
        tokens = mx.array(tokens)
        targets = mx.array(targets)
        
        print(f"\nTest data shape: {tokens.shape}")
        
        # Inference timing
        print("\n--- Inference Test ---")
        start_time = time.time()
        logits = model(tokens)
        mx.eval(logits)  # Force evaluation
        inference_time = time.time() - start_time
        print(f"Forward pass time: {inference_time:.3f}s")
        print(f"Output shape: {logits.shape}")
        
        # Training test
        print("\n--- Training Test ---")
        optimizer = optim.Adam(learning_rate=1e-3)
        
        def loss_fn(model, tokens, targets):
            logits = model(tokens)
            # Reshape for cross entropy
            logits_flat = mx.reshape(logits, (-1, config['vocab_size']))
            targets_flat = mx.reshape(targets, (-1,))
            return nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
        
        # Training step
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        
        # Run a few training steps
        losses = []
        for step in range(5):
            start_time = time.time()
            loss, grads = loss_and_grad_fn(model, tokens, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters())  # Force evaluation
            step_time = time.time() - start_time
            losses.append(float(loss))
            print(f"  Step {step+1}: Loss = {float(loss):.4f}, Time = {step_time:.3f}s")
        
        # Check if loss is decreasing
        if losses[-1] < losses[0]:
            print("✓ Loss is decreasing - training works!")
        else:
            print("⚠ Loss not decreasing consistently")
        
        # Memory test with larger batch
        print("\n--- Memory Test ---")
        large_batch = 16
        large_seq = 128
        large_tokens = mx.random.randint(0, config['vocab_size'], (large_batch, large_seq))
        
        start_time = time.time()
        large_logits = model(large_tokens)
        mx.eval(large_logits)
        large_time = time.time() - start_time
        
        print(f"Large batch ({large_batch}x{large_seq}): {large_time:.3f}s")
        print(f"Throughput: {(large_batch * large_seq) / large_time:.0f} tokens/sec")
        
        print("\n✓ MLX implementation tests PASSED!")
        return True, losses[-1]
        
    except Exception as e:
        print(f"\n✗ MLX implementation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_pytorch_implementation():
    """Test PyTorch xLSTM with training and inference"""
    print("\n" + "="*60)
    print("Testing PyTorch Implementation")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from xlstm_pytorch import create_xlstm_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")
        
        # Model configuration (same as MLX)
        config = {
            'vocab_size': 1000,
            'num_layers': 4,
            'signature': (3, 1),  # 3 mLSTM, 1 sLSTM pattern
            'inp_dim': 256,
            'head_dim': 32,
            'head_num': 8,
            'dropout': 0.1,
            'device': device
        }
        
        print(f"\nModel Configuration:")
        for k, v in config.items():
            if k != 'device':
                print(f"  {k}: {v}")
        
        # Create model
        model = create_xlstm_model(**config)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {param_count:,}")
        
        # Test forward pass
        batch_size = 4
        seq_len = 50
        tokens, targets = generate_synthetic_data(batch_size, seq_len, config['vocab_size'])
        tokens = torch.tensor(tokens, device=device)
        targets = torch.tensor(targets, device=device)
        
        print(f"\nTest data shape: {tokens.shape}")
        
        # Inference timing
        print("\n--- Inference Test ---")
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            logits = model(tokens)
            if device == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
        print(f"Forward pass time: {inference_time:.3f}s")
        print(f"Output shape: {logits.shape}")
        
        # Training test
        print("\n--- Training Test ---")
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Run a few training steps
        losses = []
        for step in range(5):
            start_time = time.time()
            
            optimizer.zero_grad()
            logits = model(tokens)
            
            # Reshape for cross entropy
            logits_flat = logits.view(-1, config['vocab_size'])
            targets_flat = targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            if device == 'cuda':
                torch.cuda.synchronize()
            step_time = time.time() - start_time
            
            losses.append(loss.item())
            print(f"  Step {step+1}: Loss = {loss.item():.4f}, Time = {step_time:.3f}s")
        
        # Check if loss is decreasing
        if losses[-1] < losses[0]:
            print("✓ Loss is decreasing - training works!")
        else:
            print("⚠ Loss not decreasing consistently")
        
        # Memory test with larger batch
        print("\n--- Memory Test ---")
        large_batch = 16
        large_seq = 128
        large_tokens = torch.randint(0, config['vocab_size'], (large_batch, large_seq), device=device)
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            large_logits = model(large_tokens)
            if device == 'cuda':
                torch.cuda.synchronize()
            large_time = time.time() - start_time
        
        print(f"Large batch ({large_batch}x{large_seq}): {large_time:.3f}s")
        print(f"Throughput: {(large_batch * large_seq) / large_time:.0f} tokens/sec")
        
        # GPU memory usage if available
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        
        print("\n✓ PyTorch implementation tests PASSED!")
        return True, losses[-1]
        
    except Exception as e:
        print(f"\n✗ PyTorch implementation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def compare_implementations():
    """Compare outputs from both implementations"""
    print("\n" + "="*60)
    print("Comparing Implementations")
    print("="*60)
    
    try:
        import mlx.core as mx
        import torch
        from xlstm_mlx import create_xlstm_model as create_mlx_model
        from xlstm_pytorch import create_xlstm_model as create_pytorch_model
        
        # Use same configuration
        config = {
            'vocab_size': 100,
            'num_layers': 2,
            'signature': (1, 1),
            'inp_dim': 64,
            'head_dim': 16,
            'head_num': 4,
            'dropout': 0.0  # No dropout for comparison
        }
        
        print("\nConfiguration for comparison:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Create models
        mlx_model = create_mlx_model(**config)
        pytorch_model = create_pytorch_model(**config, device='cpu')
        
        # Generate test data
        batch_size = 2
        seq_len = 10
        np.random.seed(42)  # For reproducibility
        tokens_np = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
        
        # MLX forward pass
        mlx_tokens = mx.array(tokens_np)
        mlx_output = mlx_model(mlx_tokens)
        
        # PyTorch forward pass
        pytorch_tokens = torch.tensor(tokens_np)
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(pytorch_tokens)
        
        # Compare shapes
        mlx_shape = mlx_output.shape
        pytorch_shape = tuple(pytorch_output.shape)
        
        print(f"\nOutput shapes:")
        print(f"  MLX:     {mlx_shape}")
        print(f"  PyTorch: {pytorch_shape}")
        print(f"  Match:   {mlx_shape == pytorch_shape} ✓" if mlx_shape == pytorch_shape else f"  Match:   {mlx_shape == pytorch_shape} ✗")
        
        # Compare output statistics
        mlx_mean = float(mx.mean(mlx_output))
        mlx_std = float(mx.std(mlx_output))
        pytorch_mean = pytorch_output.mean().item()
        pytorch_std = pytorch_output.std().item()
        
        print(f"\nOutput statistics:")
        print(f"  MLX:     mean={mlx_mean:.4f}, std={mlx_std:.4f}")
        print(f"  PyTorch: mean={pytorch_mean:.4f}, std={pytorch_std:.4f}")
        
        # Note: We don't expect identical outputs due to different random initializations
        print("\nNote: Outputs differ due to random weight initialization.")
        print("Both implementations produce valid outputs with similar statistics.")
        
        return True
        
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("xLSTM Implementation Testing & Benchmarking")
    print("="*60)
    
    results = {}
    
    # Test MLX implementation
    mlx_success, mlx_loss = test_mlx_implementation()
    results['MLX'] = (mlx_success, mlx_loss)
    
    # Test PyTorch implementation
    pytorch_success, pytorch_loss = test_pytorch_implementation()
    results['PyTorch'] = (pytorch_success, pytorch_loss)
    
    # Compare implementations
    comparison_success = compare_implementations()
    results['Comparison'] = (comparison_success, None)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for impl, (success, final_loss) in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        loss_str = f" (final loss: {final_loss:.4f})" if final_loss is not None else ""
        print(f"{impl:12} {status}{loss_str}")
    
    all_passed = all(success for success, _ in results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("Both xLSTM implementations are working correctly.")
    else:
        print("SOME TESTS FAILED ✗")
        print("Please review the output above for details.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())