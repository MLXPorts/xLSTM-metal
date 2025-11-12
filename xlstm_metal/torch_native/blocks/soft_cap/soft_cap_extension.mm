#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;
};

MetalContext& ctx() {
    static MetalContext context;
    if (!context.device) {
        context.device = MTLCreateSystemDefaultDevice();
    }
    if (!context.queue) {
        context.queue = [context.device newCommandQueue];
    }
    return context;
}

std::string load_source() {
    static const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path path = base / "soft_cap_kernel.metal";
    std::ifstream file(path);
    TORCH_CHECK(file.good(), "Failed to read Metal file: ", path.string());
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

id<MTLComputePipelineState> build_pipeline() {
    MetalContext& context = ctx();
    std::string source = load_source();
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [context.device newLibraryWithSource:src options:nil error:&error];
    TORCH_CHECK(library != nil, "Failed to compile Metal library: ", error.localizedDescription.UTF8String);
    id<MTLFunction> function = [library newFunctionWithName:@"soft_cap_kernel"];
    TORCH_CHECK(function != nil, "Missing Metal function soft_cap_kernel");
    id<MTLComputePipelineState> pipeline = [context.device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pipeline != nil, "Failed to create pipeline state: ", error.localizedDescription.UTF8String);
    return pipeline;
}

id<MTLBuffer> tensor_to_buffer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::MPS, "Tensor must be on MPS");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    return at::native::mps::getMTLBufferStorage(tensor);
}

} // namespace

torch::Tensor soft_cap_forward(torch::Tensor input, double cap_value) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "soft_cap expects float32 input");
    auto x = input.contiguous();
    auto out = torch::empty_like(x);

    MetalContext& context = ctx();
    if (!context.pso) {
        context.pso = build_pipeline();
    }

    id<MTLBuffer> in_buf = tensor_to_buffer(x);
    id<MTLBuffer> out_buf = tensor_to_buffer(out);

    float cap = static_cast<float>(cap_value);
    uint32_t numel = static_cast<uint32_t>(x.numel());
    id<MTLBuffer> cap_buf = [context.device newBufferWithBytes:&cap length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> n_buf = [context.device newBufferWithBytes:&numel length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context.pso];
    [encoder setBuffer:in_buf offset:0 atIndex:0];
    [encoder setBuffer:out_buf offset:0 atIndex:1];
    [encoder setBuffer:cap_buf offset:0 atIndex:2];
    [encoder setBuffer:n_buf offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min<NSUInteger>(context.pso.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroup = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_cap_forward", &soft_cap_forward, "SoftCap forward (MPS)");
}
