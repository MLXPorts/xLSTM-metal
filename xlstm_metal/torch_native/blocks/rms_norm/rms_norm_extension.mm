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
    std::filesystem::path path = base / "rms_norm_kernel.metal";
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
    id<MTLFunction> function = [library newFunctionWithName:@"rms_norm_kernel"];
    TORCH_CHECK(function != nil, "Missing Metal function rms_norm_kernel");
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

torch::Tensor rms_norm_forward(torch::Tensor hidden_states, torch::Tensor weight, double eps_value) {
    TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat32, "rms_norm expects float32 hidden states");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "rms_norm expects float32 weight");
    auto x = hidden_states.contiguous();
    auto w = weight.contiguous();
    TORCH_CHECK(x.dim() == 2, "hidden_states must be [batch, hidden]");
    TORCH_CHECK(w.numel() == x.size(1), "weight length must match hidden size");

    auto out = torch::empty_like(x);

    MetalContext& context = ctx();
    if (!context.pso) {
        context.pso = build_pipeline();
    }

    id<MTLBuffer> in_buf = tensor_to_buffer(x);
    id<MTLBuffer> w_buf = tensor_to_buffer(w);
    id<MTLBuffer> out_buf = tensor_to_buffer(out);

    struct RMSParams { uint32_t rows; uint32_t cols; float eps; } params;
    params.rows = static_cast<uint32_t>(x.size(0));
    params.cols = static_cast<uint32_t>(x.size(1));
    params.eps = static_cast<float>(eps_value);

    id<MTLBuffer> params_buf = [context.device newBufferWithBytes:&params length:sizeof(RMSParams) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context.pso];
    [encoder setBuffer:in_buf offset:0 atIndex:0];
    [encoder setBuffer:w_buf offset:0 atIndex:1];
    [encoder setBuffer:out_buf offset:0 atIndex:2];
    [encoder setBuffer:params_buf offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake(params.rows, 1, 1);
    MTLSize threadgroup = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_forward", &rms_norm_forward, "RMSNorm forward (MPS)");
}
