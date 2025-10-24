#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

// Helper function to get MTLBuffer from PyTorch tensor
id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static id<MTLLibrary> _compileLibraryFromSource(id<MTLDevice> device, const std::string &src) {
    NSString *shaderSource = [NSString stringWithUTF8String:src.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        throw std::runtime_error("Failed to compile Metal library from provided source");
    }
    return library;
}

torch::Tensor metal_soft_cap_with_source(torch::Tensor input, float cap_value, const std::string &shader_src) {
    TORCH_CHECK(input.device().type() == torch::kMPS, "Input must be on MPS device");
    
    // Get Metal device and command queue
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
    
    // Compile provided Metal shader source
    id<MTLLibrary> library = _compileLibraryFromSource(metalDevice, shader_src);
    NSError* error = nil;
    
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"soft_cap_kernel"];
    if (!kernelFunction) {
        throw std::runtime_error("Metal: function 'soft_cap_kernel' not found in compiled library");
    }
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

std::vector<torch::Tensor> metal_mlstm_step_with_source(
    torch::Tensor q,      // [B, NH, DHQK]
    torch::Tensor k,      // [B, NH, DHQK]
    torch::Tensor v,      // [B, NH, DHHV]
    torch::Tensor i_pre,  // [B, NH]
    torch::Tensor f_pre,  // [B, NH]
    torch::Tensor C,      // [B, NH, DHQK, DHHV] (in-place update)
    torch::Tensor N,      // [B, NH, DHQK] (in-place update)
    torch::Tensor M,      // [B, NH] (in-place update)
    float eps,
    const std::string &shader_src
) {
    TORCH_CHECK(q.device().type() == torch::kMPS, "Inputs must be on MPS device");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(C.scalar_type() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(N.scalar_type() == torch::kFloat32, "N must be float32");
    TORCH_CHECK(M.scalar_type() == torch::kFloat32, "M must be float32");
    auto sizes = q.sizes();
    TORCH_CHECK(q.dim() == 3, "q must be [B,NH,DHQK]");
    int64_t B = sizes[0], NH = sizes[1], DHQK = sizes[2];
    TORCH_CHECK(k.sizes() == sizes, "k must match q shape");
    TORCH_CHECK(v.dim() == 3 && v.sizes()[0] == B && v.sizes()[1] == NH, "v shape mismatch");
    int64_t DHHV = v.sizes()[2];

    // Prepare output h
    auto H = torch::empty_like(v);

    // Get Metal device and command queue
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];

    // Compile provided Metal shader source
    NSError* error = nil;
    id<MTLLibrary> library = _compileLibraryFromSource(metalDevice, shader_src);
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"mlstm_step_full_kernel"];
    if (!kernelFunction) {
        throw std::runtime_error("Metal: function 'mlstm_step_full_kernel' not found in compiled library");
    }
    id<MTLComputePipelineState> computePipelineState = [metalDevice newComputePipelineStateWithFunction:kernelFunction error:&error];

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:computePipelineState];

    // Ensure contiguous tensors
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
    C = C.contiguous(); N = N.contiguous(); M = M.contiguous();

    id<MTLBuffer> qBuf = getMTLBufferStorage(q);
    id<MTLBuffer> kBuf = getMTLBufferStorage(k);
    id<MTLBuffer> vBuf = getMTLBufferStorage(v);
    id<MTLBuffer> iBuf = getMTLBufferStorage(i_pre);
    id<MTLBuffer> fBuf = getMTLBufferStorage(f_pre);
    id<MTLBuffer> CBuf = getMTLBufferStorage(C);
    id<MTLBuffer> NBuf = getMTLBufferStorage(N);
    id<MTLBuffer> MBuf = getMTLBufferStorage(M);
    id<MTLBuffer> HBuf = getMTLBufferStorage(H);

    [encoder setBuffer:qBuf offset:0 atIndex:0];
    [encoder setBuffer:kBuf offset:0 atIndex:1];
    [encoder setBuffer:vBuf offset:0 atIndex:2];
    [encoder setBuffer:iBuf offset:0 atIndex:3];
    [encoder setBuffer:fBuf offset:0 atIndex:4];
    [encoder setBuffer:CBuf offset:0 atIndex:5];
    [encoder setBuffer:NBuf offset:0 atIndex:6];
    [encoder setBuffer:MBuf offset:0 atIndex:7];
    [encoder setBuffer:HBuf offset:0 atIndex:8];

    uint32_t B32 = (uint32_t)B;
    uint32_t NH32 = (uint32_t)NH;
    uint32_t DHQK32 = (uint32_t)DHQK;
    uint32_t DHHV32 = (uint32_t)DHHV;
    float eps32 = eps;
    [encoder setBytes:&B32 length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&NH32 length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&DHQK32 length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&DHHV32 length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&eps32 length:sizeof(float) atIndex:13];

    MTLSize gridSize = MTLSizeMake((NSUInteger)DHHV, (NSUInteger)NH, (NSUInteger)B);
    NSUInteger tpt = computePipelineState.maxTotalThreadsPerThreadgroup;
    if (tpt > 256) tpt = 256;
    // Choose a 1D threadgroup along DHHV
    NSUInteger tgx = (NSUInteger) (tpt);
    if (tgx > DHHV) tgx = (NSUInteger)DHHV;
    MTLSize tgSize = MTLSizeMake(tgx, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    return {H, C, N, M};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("metal_soft_cap_with_source", &metal_soft_cap_with_source, "Metal soft cap (source provided)");
    m.def("metal_mlstm_step_with_source", &metal_mlstm_step_with_source, "Metal mLSTM step (source provided)");
    m.def("metal_memcpy_with_source", [](torch::Tensor input, const std::string &shader_src){
        TORCH_CHECK(input.device().type() == torch::kMPS, "Input must be on MPS device");
        id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = _compileLibraryFromSource(metalDevice, shader_src);
        id<MTLFunction> fn = [library newFunctionWithName:@"memcpy_kernel"];
        if (!fn) throw std::runtime_error("Metal: function 'memcpy_kernel' not found");
        id<MTLComputePipelineState> pso = [metalDevice newComputePipelineStateWithFunction:fn error:&error];
        auto output = torch::empty_like(input);
        id<MTLBuffer> inBuf = getMTLBufferStorage(input);
        id<MTLBuffer> outBuf = getMTLBufferStorage(output);
        id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:inBuf offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];
        uint32_t size = (uint32_t)input.numel();
        [enc setBytes:&size length:sizeof(uint32_t) atIndex:2];
        MTLSize grid = MTLSizeMake(size,1,1);
        NSUInteger tpt = pso.maxTotalThreadsPerThreadgroup;
        if (tpt > size) tpt = size;
        MTLSize tg = MTLSizeMake(tpt,1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return output;
    }, "Metal memcpy (source provided)");
}
