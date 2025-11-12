#include <metal_stdlib>
using namespace metal;

struct RMSParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rms_norm_kernel(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant RMSParams& params  [[buffer(3)]],
    uint row                    [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint offset = row * params.cols;
    float mean_square = 0.0f;
    for (uint i = 0; i < params.cols; ++i) {
        float val = input[offset + i];
        mean_square += val * val;
    }
    mean_square /= (float)params.cols;
    float inv_rms = rsqrt(mean_square + params.eps);
    for (uint i = 0; i < params.cols; ++i) {
        float val = input[offset + i] * inv_rms;
        output[offset + i] = val * weight[i];
    }
}
