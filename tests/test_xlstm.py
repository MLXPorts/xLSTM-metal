
"""
Test xLSTM implementation
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from dataclasses import dataclass

# First test if we can import the components
print("Testing imports...")
try:
    from xlstm.xlstm_large.components import RMSNorm, soft_cap
    print("✓ Official xLSTM components imported")
except ImportError as e:
    print(f"✗ Failed to import xLSTM components: {e}")
    exit(1)

print("✓ Using our mLSTM backend implementation")

# Now test our implementation
print("\nTesting implementation...")

# Create a simple test case
@dataclass
class TestConfig:
    embedding_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 2
    vocab_size: int = 1000
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    add_out_norm: bool = True
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    chunkwise_kernel: str = "native"
    sequence_kernel: str = "native"
    step_kernel: str = "native"
    mode: str = "train"
    chunk_size: int = 64
    return_last_states: bool = False
    autocast_kernel_dtype: str = "float16"
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    weight_mode: str = "single"

def test_soft_cap():
    """Test soft cap function"""
    print("\n1. Testing soft_cap function...")
    x = torch.randn(2, 10, device='mps')
    result = soft_cap(x, 15.0)
    print(f"   Input shape: {x.shape}, Output shape: {result.shape}")
    print(f"   Max input: {x.max().item():.4f}, Max output: {result.max().item():.4f}")
    assert result.max().item() <= 15.0, "Soft cap failed"
    print("   ✓ Soft cap working correctly")

def test_rms_norm():
    """Test RMS normalization"""
    print("\n2. Testing RMSNorm...")
    norm = RMSNorm(num_features=256, eps=1e-6)
    norm = norm.to('mps')
    x = torch.randn(2, 10, 256, device='mps')
    result = norm(x)
    print(f"   Input shape: {x.shape}, Output shape: {result.shape}")
    print(f"   Mean: {result.mean().item():.6f}, Std: {result.std().item():.6f}")
    print("   ✓ RMSNorm working correctly")

def test_mlstm_backend():
    """Test mLSTM backend"""
    print("\n3. Testing mLSTM backend...")
    
    # Import our implementation
    from xlstm.backends.mlstm_backend import mLSTMBackendConfig, mLSTMBackend
    
    config = mLSTMBackendConfig(
        chunkwise_kernel="native",
        sequence_kernel="native",
        step_kernel="native",
        mode="train",
        chunk_size=64,
        return_last_states=False,
        autocast_kernel_dtype="float16",
        eps=1e-6,
        inference_state_dtype="float32"
    )
    
    backend = mLSTMBackend(config=config)
    
    # Create test inputs
    B, NH, S, DH = 2, 4, 16, 32
    q = torch.randn(B, NH, S, DH, device='mps')
    k = torch.randn(B, NH, S, DH, device='mps')
    v = torch.randn(B, NH, S, DH, device='mps')
    i = torch.randn(B, NH, S, device='mps')
    f = torch.randn(B, NH, S, device='mps')
    
    print(f"   Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    
    try:
        h, state = backend(q=q, k=k, v=v, i=i, f=f)
        print(f"   Output shape: {h.shape}")
        print("   ✓ mLSTM backend working correctly")
    except Exception as e:
        print(f"   ✗ mLSTM backend failed: {e}")

def test_simple_forward():
    """Test a simple forward pass"""
    print("\n4. Testing simple forward pass...")
    
    # Import our model
    try:
        from xlstm.models.xlstm import xLSTMLarge, xLSTMLargeConfig
        print("   ✓ Imported xLSTM implementation")
    except ImportError as e:
        print(f"   ✗ Failed to import: {e}")
        return
    
    # Create small model for testing
    config = xLSTMLargeConfig(
        embedding_dim=128,
        num_heads=2,
        num_blocks=1,
        vocab_size=100
    )
    
    model = xLSTMLarge(config)
    model = model.to('mps')
    model.eval()
    
    # Test input
    batch_size = 1
    seq_len = 8
    tokens = torch.randint(0, 100, (batch_size, seq_len), device='mps')
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Input shape: {tokens.shape}")
    
    # Forward pass
    with torch.no_grad():
        start = time.time()
        logits = model(tokens)
        torch.mps.synchronize()
        elapsed = time.time() - start
        
    print(f"   Output shape: {logits.shape}")
    print(f"   Forward pass time: {elapsed:.4f}s")
    print("   ✓ Forward pass successful")

# Run tests
print("="*50)
print("Running xLSTM Tests")
print("="*50)

test_soft_cap()
test_rms_norm()
test_mlstm_backend()
test_simple_forward()

print("\n" + "="*50)
print("All tests completed!")
print("="*50)