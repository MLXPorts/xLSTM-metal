
# Archived demo: moved from repo root; not used by production backends.
"""
PyTorch Metal Kernel Implementation for xLSTM

Implementation of custom Metal kernels for PyTorch MPS backend,
based on research from:
- pytorch-cpp-metal-tutorial by smrfeld
- Custom PyTorch Operations for Metal Backend (Medium)
- PyTorch MPS backend documentation

This demonstrates how to bridge our MLX Metal kernels to PyTorch.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import os
from pathlib import Path


# Check if Metal Performance Shaders is available
if not torch.backends.mps.is_available():
    raise RuntimeError("Metal Performance Shaders (MPS) not available")

device = torch.device("mps")


def create_pytorch_metal_extension():
    """
    Creates the C++ extension for PyTorch Metal kernels.
    
    Based on research, this requires:
    1. Metal shader files (.metal)
    2. C++/Objective-C++ implementation (.mm files) 
    3. Python wrapper with pybind11
    4. Custom setup.py with Metal framework linking
    """
    
    # This would be the structure for a complete implementation:
    extension_files = {
        'setup.py': '''
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import pkg_resources

# Custom build class to handle Metal files
class MetalBuildExt(build_ext):
    def build_extensions(self):
        # Add Metal framework linking
        for ext in self.extensions:
            ext.extra_link_args += ['-framework', 'Metal', '-framework', 'Foundation']
        super().build_extensions()

ext_modules = [
    Pybind11Extension(
        "pytorch_metal_xlstm",
        ["pytorch_metal_xlstm.mm"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": MetalBuildExt},
    zip_safe=False,
)
''',
        
        'pytorch_metal_xlstm.mm': '''
#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

// Helper function to get MTLBuffer from PyTorch tensor
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor metal_soft_cap_pytorch(torch::Tensor input, float cap_value) {
    TORCH_CHECK(input.device().type() == torch::kMPS, "Input must be on MPS device");
    
    // Get Metal device and command queue
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
    
    // Load and compile Metal shader
    NSString* shaderSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void soft_cap_kernel(
            device float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant float& cap_value [[buffer(2)]],
            constant uint& size [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= size) return;
            float val = input[id];
            output[id] = cap_value * tanh(val / cap_value);
        }
    )";
    
    NSError* error = nil;
    id<MTLLibrary> library = [metalDevice newLibraryWithSource:shaderSource 
                                                       options:nil 
                                                         error:&error];
    if (!library) {
        throw std::runtime_error("Failed to compile Metal library");
    }
    
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"soft_cap_kernel"];
    id<MTLComputePipelineState> computePipelineState = 
        [metalDevice newComputePipelineStateWithFunction:kernelFunction error:&error];
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Get Metal buffers
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(input);
    id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:computePipelineState];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBytes:&cap_value length:sizeof(float) atIndex:2];
    uint32_t size = input.numel();
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:3];
    
    // Configure thread groups
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = computePipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > size) threadGroupSize = size;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("metal_soft_cap", &metal_soft_cap_pytorch, "Metal soft cap implementation");
}
''',
        
        'xlstm_kernels.metal': '''
#include <metal_stdlib>
using namespace metal;

kernel void soft_cap_kernel(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& cap_value [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= size) return;
    float val = input[id];
    output[id] = cap_value * tanh(val / cap_value);
}

kernel void mlstm_step_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    device float* i_gate [[buffer(3)]],
    device float* f_gate [[buffer(4)]],
    device float* o_gate [[buffer(5)]],
    device float* hidden_state [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint3 id [[thread_position_in_grid]]
) {
    uint batch = id.z;
    uint head = id.y;  
    uint dim = id.x;
    
    if (batch >= batch_size || head >= num_heads || dim >= head_dim) return;
    
    uint idx = batch * num_heads * head_dim + head * head_dim + dim;
    uint gate_idx = batch * num_heads + head;
    
    float q_val = q[idx];
    float k_val = k[idx];
    float v_val = v[idx];
    float i_val = i_gate[gate_idx];
    float f_val = f_gate[gate_idx];
    float o_val = o_gate[gate_idx];
    
    // Update matrix memory: H = f * H + i * (k ⊗ v)
    uint h_base = batch * num_heads * head_dim * head_dim + head * head_dim * head_dim;
    
    // Compute outer product k ⊗ v and update memory
    for (uint j = 0; j < head_dim; j++) {
        uint h_idx = h_base + dim * head_dim + j;
        float kv_outer = k_val * v[batch * num_heads * head_dim + head * head_dim + j];
        hidden_state[h_idx] = f_val * hidden_state[h_idx] + i_val * kv_outer;
    }
    
    // Compute output: h = H * q
    float h_sum = 0.0f;
    for (uint j = 0; j < head_dim; j++) {
        uint h_idx = h_base + j * head_dim + dim;
        h_sum += hidden_state[h_idx] * q_val;
    }
    
    // Apply output gate
    output[idx] = o_val * h_sum;
}
'''
    }
    
    return extension_files


class PyTorchMetalSoftCap(nn.Module):
    """
    PyTorch implementation using fallback to MPS operations
    until custom Metal extension is built
    """
    
    def __init__(self, cap_value: float = 15.0):
        super().__init__()
        self.cap_value = cap_value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use MPS-optimized operations as fallback
        return self.cap_value * torch.tanh(x / self.cap_value)


class PyTorchMetalRMSNorm(nn.Module):
    """RMSNorm optimized for MPS backend"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MPS-optimized implementation
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class PyTorchMetalmLSTMBlock(nn.Module):
    """mLSTM block optimized for PyTorch MPS backend"""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        
        # Gates with Metal soft capping
        self.i_proj = nn.Linear(d_model, num_heads, bias=False)
        self.f_proj = nn.Linear(d_model, num_heads, bias=False)
        self.o_proj = nn.Linear(d_model, num_heads, bias=False)
        
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.soft_cap = PyTorchMetalSoftCap(15.0)
        self.layer_norm = PyTorchMetalRMSNorm(d_model)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        residual = x
        x = self.layer_norm(x)
        
        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Gates with soft capping
        i_gate = torch.sigmoid(self.soft_cap(self.i_proj(x)))
        f_gate = torch.sigmoid(self.soft_cap(self.f_proj(x)))
        o_gate = torch.sigmoid(self.soft_cap(self.o_proj(x)))
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )
        
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t]  # [batch_size, num_heads, head_dim]
            k_t = k[:, t]
            v_t = v[:, t]
            i_t = i_gate[:, t]  # [batch_size, num_heads]
            f_t = f_gate[:, t]
            o_t = o_gate[:, t]
            
            # Matrix memory update (optimized for MPS)
            kv_outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)
            hidden_state = f_t.unsqueeze(-1).unsqueeze(-1) * hidden_state + i_t.unsqueeze(-1).unsqueeze(-1) * kv_outer
            
            # Compute output
            h_t = torch.einsum('bhd,bhde->bhe', q_t, hidden_state)
            h_t = o_t.unsqueeze(-1) * h_t
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)
        output = output.view(batch_size, seq_len, -1)
        
        return residual + self.out_proj(output), hidden_state


class PyTorchMetalxLSTMModel(nn.Module):
    """Complete xLSTM model optimized for PyTorch MPS backend"""
    
    def __init__(self, vocab_size: int = 50257, num_layers: int = 6, d_model: int = 512):
        super().__init__()
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # xLSTM blocks optimized for MPS
        self.blocks = nn.ModuleList([
            PyTorchMetalmLSTMBlock(d_model=d_model)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.soft_cap = PyTorchMetalSoftCap(30.0)
        
    def forward(self, tokens: torch.Tensor, hidden_states: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        x = self.embedding(tokens)
        
        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)
        
        for i, block in enumerate(self.blocks):
            x, hidden_states[i] = block(x, hidden_states[i])
        
        logits = self.head(x)
        
        # Apply soft capping to output logits
        logits = self.soft_cap(logits)
        
        return logits, hidden_states


def build_metal_extension():
    """
    Instructions for building the complete Metal extension.
    
    This function provides the template and instructions for creating
    a full C++/Metal extension for PyTorch.
    """
    
    instructions = """
    To build the complete Metal extension:
    
    1. Create extension directory structure:
       mkdir pytorch_metal_xlstm_ext
       cd pytorch_metal_xlstm_ext
    
    2. Create files from create_pytorch_metal_extension()
    
    3. Install with:
       pip install -e .
    
    4. Usage:
       import pytorch_metal_xlstm
       result = pytorch_metal_xlstm.metal_soft_cap(tensor, cap_value)
    
    Key challenges (based on research):
    - Metal libraries can't be linked with object files
    - PyTorch hasn't exposed MPS APIs officially
    - Need to patch build system for Metal support
    - Requires copying headers from PyTorch csrc
    
    Recommended approach: JIT-compile Metal kernels at runtime
    """
    
    return instructions


def create_metal_kernel_files():
    """Create the Metal kernel files for the extension"""
    
    # Create directory
    os.makedirs("pytorch_metal_xlstm_ext", exist_ok=True)
    
    # Write extension files
    extension_files = create_pytorch_metal_extension()
    
    for filename, content in extension_files.items():
        with open(f"pytorch_metal_xlstm_ext/{filename}", "w") as f:
            f.write(content)
    
    print("Created PyTorch Metal extension files in pytorch_metal_xlstm_ext/")
    print("To build: cd pytorch_metal_xlstm_ext && pip install -e .")


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing PyTorch Metal xLSTM implementation...")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Create model
    model = PyTorchMetalxLSTMModel(
        vocab_size=1000,
        num_layers=4,
        d_model=256
    ).to(device)
    
    # Test data
    batch_size = 1
    seq_len = 32
    prompt = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    print(f"Prompt shape: {prompt.shape}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        logits, hidden_states = model(prompt)
    torch.mps.synchronize()  # Ensure MPS operations complete
    forward_time = time.time() - start_time
    
    print(f"Forward pass completed in {forward_time:.3f}s")
    print(f"Output shape: {logits.shape}")
    
    # Test soft capping
    test_tensor = torch.randn(100, device=device) * 10
    soft_cap = PyTorchMetalSoftCap(5.0)
    capped = soft_cap(test_tensor)
    
    print(f"Soft capping: max uncapped = {test_tensor.max():.2f}, max capped = {capped.max():.2f}")
    
    # Create Metal extension files
    create_metal_kernel_files()
    
    print("\n" + build_metal_extension())
    
    print("PyTorch Metal xLSTM implementation complete!")
