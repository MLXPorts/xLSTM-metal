#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace py = pybind11;

namespace {

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
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
    std::filesystem::path path = base / "mlstm_chunkwise_recurrent_fw_C_kernel.metal";
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
    id<MTLFunction> function = [library newFunctionWithName:@"mlstm_recurrent_fw_C"];
    TORCH_CHECK(function != nil, "Missing Metal function mlstm_recurrent_fw_C");
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

void mlstm_recurrent_fw_C_forward(
    torch::Tensor matK,
    torch::Tensor matV,
    torch::Tensor vecF,
    torch::Tensor vecI,
    torch::Tensor matC_initial,
    torch::Tensor vecN_initial,
    torch::Tensor scaMinter_initial,
    torch::Tensor matC_states,
    torch::Tensor vecN_states,
    torch::Tensor scaMinter_states,
    torch::Tensor dbg,
    torch::Tensor params,
    torch::Tensor strides,
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    int64_t threadgroup_x,
    int64_t threadgroup_y
) {
    TORCH_CHECK(matK.scalar_type() == torch::kFloat32, "matK must be float32");
    TORCH_CHECK(matV.scalar_type() == torch::kFloat32, "matV must be float32");
    TORCH_CHECK(vecF.scalar_type() == torch::kFloat32, "vecF must be float32");
    TORCH_CHECK(vecI.scalar_type() == torch::kFloat32, "vecI must be float32");
    TORCH_CHECK(matC_states.scalar_type() == torch::kFloat32 || matC_states.scalar_type() == torch::kFloat16,
                "matC_states must be float");

    auto matK_ = matK.contiguous();
    auto matV_ = matV.contiguous();
    auto vecF_ = vecF.contiguous();
    auto vecI_ = vecI.contiguous();
    auto matC_init = matC_initial.contiguous();
    auto vecN_init = vecN_initial.contiguous();
    auto scaMinter_init = scaMinter_initial.contiguous();
    auto matC_out = matC_states.contiguous();
    auto vecN_out = vecN_states.contiguous();
    auto scaMinter_out = scaMinter_states.contiguous();
    auto dbg_out = dbg.contiguous();
    auto params_ = params.contiguous();
    auto strides_ = strides.contiguous();

    MetalContext& context = ctx();
    if (!context.pipeline) {
        context.pipeline = build_pipeline();
    }

    id<MTLBuffer> buf_matK = tensor_to_buffer(matK_);
    id<MTLBuffer> buf_matV = tensor_to_buffer(matV_);
    id<MTLBuffer> buf_vecF = tensor_to_buffer(vecF_);
    id<MTLBuffer> buf_vecI = tensor_to_buffer(vecI_);
    id<MTLBuffer> buf_matC_init = tensor_to_buffer(matC_init);
    id<MTLBuffer> buf_vecN_init = tensor_to_buffer(vecN_init);
    id<MTLBuffer> buf_scaMinter_init = tensor_to_buffer(scaMinter_init);
    id<MTLBuffer> buf_matC_out = tensor_to_buffer(matC_out);
    id<MTLBuffer> buf_vecN_out = tensor_to_buffer(vecN_out);
    id<MTLBuffer> buf_scaMinter_out = tensor_to_buffer(scaMinter_out);
    id<MTLBuffer> buf_dbg = tensor_to_buffer(dbg_out);
    id<MTLBuffer> buf_params = tensor_to_buffer(params_);
    id<MTLBuffer> buf_strides = tensor_to_buffer(strides_);

    id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:context.pipeline];
    [encoder setBuffer:buf_matK offset:0 atIndex:0];
    [encoder setBuffer:buf_matV offset:0 atIndex:1];
    [encoder setBuffer:buf_vecF offset:0 atIndex:2];
    [encoder setBuffer:buf_vecI offset:0 atIndex:3];
    [encoder setBuffer:buf_matC_init offset:0 atIndex:4];
    [encoder setBuffer:buf_vecN_init offset:0 atIndex:5];
    [encoder setBuffer:buf_scaMinter_init offset:0 atIndex:6];
    [encoder setBuffer:buf_matC_out offset:0 atIndex:7];
    [encoder setBuffer:buf_vecN_out offset:0 atIndex:8];
    [encoder setBuffer:buf_scaMinter_out offset:0 atIndex:9];
    [encoder setBuffer:buf_dbg offset:0 atIndex:10];
    [encoder setBuffer:buf_params offset:0 atIndex:11];
    [encoder setBuffer:buf_strides offset:0 atIndex:12];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(grid_x),
                               static_cast<NSUInteger>(grid_y),
                               static_cast<NSUInteger>(grid_z));
    MTLSize threads = MTLSizeMake(static_cast<NSUInteger>(threadgroup_x),
                                  static_cast<NSUInteger>(threadgroup_y),
                                  1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "mlstm_recurrent_fw_C_forward",
        &mlstm_recurrent_fw_C_forward,
        "Run the mLSTM chunkwise recurrent forward Metal kernel",
        py::arg("matK"),
        py::arg("matV"),
        py::arg("vecF"),
        py::arg("vecI"),
        py::arg("matC_initial"),
        py::arg("vecN_initial"),
        py::arg("scaMinter_initial"),
        py::arg("matC_states"),
        py::arg("vecN_states"),
        py::arg("scaMinter_states"),
        py::arg("dbg"),
        py::arg("params"),
        py::arg("strides"),
        py::arg("grid_x"),
        py::arg("grid_y"),
        py::arg("grid_z"),
        py::arg("threadgroup_x"),
        py::arg("threadgroup_y")
    );
}
