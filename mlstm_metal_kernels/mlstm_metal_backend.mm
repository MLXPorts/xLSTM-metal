
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
