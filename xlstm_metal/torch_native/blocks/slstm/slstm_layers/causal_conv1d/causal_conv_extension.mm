#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

struct Pipelines {
    id<MTLComputePipelineState> mixing = nil;
    id<MTLComputePipelineState> depthwise = nil;
};

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    Pipelines pipelines;
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

std::string load_source(const char* filename) {
    static const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path path = base / filename;
    std::ifstream file(path);
    TORCH_CHECK(file.good(), "Failed to read Metal file: ", path.string());
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

id<MTLComputePipelineState> build_pipeline(const char* filename, const char* func_name) {
    MetalContext& context = ctx();
    std::string source = load_source(filename);
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [context.device newLibraryWithSource:src options:nil error:&error];
    TORCH_CHECK(library != nil, "Failed to compile Metal library: ", error.localizedDescription.UTF8String);
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:func_name]];
    TORCH_CHECK(function != nil, "Missing Metal function: ", func_name);
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

torch::Tensor causal_conv1d_mixing(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    auto input = x.contiguous();
    auto w = weight.contiguous();
    TORCH_CHECK(input.dim() == 3, "x must be [B, S, C]");
    TORCH_CHECK(w.dim() == 3, "weight must be [C, C, K]");
    auto bias = bias_opt.has_value() ? bias_opt.value().contiguous() : torch::zeros({input.size(2)}, input.options());
    auto output = torch::empty_like(input);

    MetalContext& context = ctx();
    if (!context.pipelines.mixing) {
        context.pipelines.mixing = build_pipeline("mixing_kernel.metal", "causal_conv1d_mixing");
    }

    struct MixingParams { uint32_t B, S, C, K, has_bias; } params;
    params.B = static_cast<uint32_t>(input.size(0));
    params.S = static_cast<uint32_t>(input.size(1));
    params.C = static_cast<uint32_t>(input.size(2));
    params.K = static_cast<uint32_t>(w.size(2));
    params.has_bias = bias_opt.has_value() ? 1u : 0u;

    id<MTLBuffer> param_buf = [context.device newBufferWithBytes:&params length:sizeof(MixingParams) options:MTLResourceStorageModeShared];
    id<MTLBuffer> in_buf = tensor_to_buffer(input);
    id<MTLBuffer> w_buf = tensor_to_buffer(w);
    id<MTLBuffer> b_buf = tensor_to_buffer(bias);
    id<MTLBuffer> out_buf = tensor_to_buffer(output);

    uint32_t total = params.B * params.S * params.C;

    id<MTLCommandBuffer> cb = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:context.pipelines.mixing];
    [enc setBuffer:in_buf offset:0 atIndex:0];
    [enc setBuffer:w_buf offset:0 atIndex:1];
    [enc setBuffer:b_buf offset:0 atIndex:2];
    [enc setBuffer:param_buf offset:0 atIndex:3];
    [enc setBuffer:out_buf offset:0 atIndex:4];
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min<NSUInteger>(context.pipelines.mixing.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroup = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return output;
}

torch::Tensor causal_conv1d_depthwise(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    auto input = x.contiguous();
    auto w = weight.contiguous();
    TORCH_CHECK(w.dim() == 2, "weight must be [C, K]");
    auto bias = bias_opt.has_value() ? bias_opt.value().contiguous() : torch::zeros({input.size(2)}, input.options());
    auto output = torch::empty_like(input);

    MetalContext& context = ctx();
    if (!context.pipelines.depthwise) {
        context.pipelines.depthwise = build_pipeline("depthwise_kernel.metal", "causal_conv1d_depthwise");
    }

    struct DepthParams { uint32_t B, S, C, K, has_bias; } params;
    params.B = static_cast<uint32_t>(input.size(0));
    params.S = static_cast<uint32_t>(input.size(1));
    params.C = static_cast<uint32_t>(input.size(2));
    params.K = static_cast<uint32_t>(w.size(1));
    params.has_bias = bias_opt.has_value() ? 1u : 0u;

    id<MTLBuffer> param_buf = [context.device newBufferWithBytes:&params length:sizeof(DepthParams) options:MTLResourceStorageModeShared];
    id<MTLBuffer> in_buf = tensor_to_buffer(input);
    id<MTLBuffer> w_buf = tensor_to_buffer(w);
    id<MTLBuffer> b_buf = tensor_to_buffer(bias);
    id<MTLBuffer> out_buf = tensor_to_buffer(output);

    uint32_t total = params.B * params.S * params.C;

    id<MTLCommandBuffer> cb = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:context.pipelines.depthwise];
    [enc setBuffer:in_buf offset:0 atIndex:0];
    [enc setBuffer:w_buf offset:0 atIndex:1];
    [enc setBuffer:b_buf offset:0 atIndex:2];
    [enc setBuffer:param_buf offset:0 atIndex:3];
    [enc setBuffer:out_buf offset:0 atIndex:4];
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min<NSUInteger>(context.pipelines.depthwise.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroup = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mixing_forward", &causal_conv1d_mixing, "Causal Conv1d mixing (MPS)");
    m.def("depthwise_forward", &causal_conv1d_depthwise, "Causal Conv1d depthwise (MPS)");
}
