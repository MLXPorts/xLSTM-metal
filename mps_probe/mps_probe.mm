#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Optional internal headers (may not be available in binary wheels)
#if __has_include(<ATen/mps/MPSStream.h>)
#  include <ATen/mps/MPSStream.h>
#  define HAVE_MPS_STREAM 1
#else
#  define HAVE_MPS_STREAM 0
#endif

static id<MTLLibrary> compile_library_from_source(id<MTLDevice> device, const std::string &src) {
    NSString *shaderSource = [NSString stringWithUTF8String:src.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        throw std::runtime_error("Failed to compile Metal library from provided source");
    }
    return library;
}

static const char* kMemcpyKernel = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void memcpy_kernel(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& size [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= size) return;
        output[id] = input[id];
    }
)";

// Attempt 1: standalone queue + bitcast (expected to fail on MPS tensors)
static bool attempt1_direct_bitcast(torch::Tensor input, torch::Tensor output, std::string &log) {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLLibrary> lib = compile_library_from_source(device, kMemcpyKernel);
        NSError* error = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"memcpy_kernel"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];

        // Bitcast underlying storage pointer (known-bad path)
        const void* in_ptr_c = input.storage().data();
        const void* out_ptr_c = output.storage().data();
        void* in_ptr = const_cast<void*>(in_ptr_c);
        void* out_ptr = const_cast<void*>(out_ptr_c);
        NSUInteger nbytes = (NSUInteger)(input.numel() * input.element_size());

        id<MTLBuffer> inBuf = (__bridge id<MTLBuffer>)(in_ptr);
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)(out_ptr);
        if (!inBuf || !outBuf) {
            log += "bitcast produced nil MTLBuffer\n";
            return false;
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
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
        return true; // will validate in caller
    } catch (const std::exception &e) {
        log += std::string("attempt1 exception: ") + e.what() + "\n";
        return false;
    }
}

// Attempt 2: encode inside PyTorch's MPS stream/command buffer if available
static bool attempt2_mps_stream(torch::Tensor input, torch::Tensor output, std::string &log) {
#if HAVE_MPS_STREAM
    try {
        using namespace at::mps;
        MPSStream* stream = getCurrentMPSStream();
        id<MTLCommandQueue> queue = stream->commandQueue();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> lib = compile_library_from_source(device, kMemcpyKernel);
        NSError* error = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"memcpy_kernel"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];

        // NOTE: Without a public API to get MTLBuffer from tensor, we cannot bind buffers here.
        // This section exists to demonstrate encoding within the MPS stream.
        // If PyTorch exposes buffer/texture handles in the future, bind them here before dispatching.
        (void)pso; (void)cmd;
        log += "HAVE_MPS_STREAM present but no public tensor->MTLBuffer API\n";
        return false;
    } catch (const std::exception &e) {
        log += std::string("attempt2 exception: ") + e.what() + "\n";
        return false;
    }
#else
    log += "MPSStream headers not available in this build.\n";
    return false;
#endif
}

static void log_tensor_details(const torch::Tensor &t, const char* name) {
    std::cout << name << " details:\n";
    std::cout << "  device: " << t.device() << " dtype: " << t.dtype() << "\n";
    std::cout << "  is_contiguous: " << t.is_contiguous() << "\n";
    std::cout << "  sizes: ["; for (auto s : t.sizes()) std::cout << s << " "; std::cout << "]\n";
    std::cout << "  strides: ["; for (auto s : t.strides()) std::cout << s << " "; std::cout << "]\n";
    std::cout << "  storage_offset: " << t.storage_offset() << "\n";
    const void* ptr = t.storage().data();
    std::cout << "  data ptr: " << ptr << " align%16=" << ((uintptr_t)ptr % 16) << "\n";
}

torch::Tensor debug_memcpy_mps(torch::Tensor input) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::MPS, "input must be MPS tensor");
    input = input.contiguous();
    auto output = torch::empty_like(input);

    std::string log;
    log_tensor_details(input, "input");
    log_tensor_details(output, "output");

    std::cout << "\n=== ATTEMPT 1: Direct bitcast ===\n";
    bool ok1 = attempt1_direct_bitcast(input, output, log);
    std::cout << "attempt1 returned: " << ok1 << "\n";

    std::cout << "\n=== ATTEMPT 2: MPS stream integration ===\n";
    bool ok2 = attempt2_mps_stream(input, output, log);
    std::cout << "attempt2 returned: " << ok2 << "\n";

    std::cout << log;
    return output;
}

TORCH_LIBRARY(xlstm_mps_probe, m) {
    m.def("debug_memcpy(Tensor input) -> Tensor", &debug_memcpy_mps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Python module init to satisfy import; ops are registered via TORCH_LIBRARY
}
