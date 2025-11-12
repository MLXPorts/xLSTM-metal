#include <metal_stdlib>
using namespace metal;

struct MixingParams {
    uint B;
    uint S;
    uint C;
    uint K;
    uint has_bias;
};

kernel void causal_conv1d_mixing(
    device const float* x         [[buffer(0)]],
    device const float* weight    [[buffer(1)]],
    device const float* bias      [[buffer(2)]],
    constant MixingParams& params [[buffer(3)]],
    device float* output          [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]) {
    uint total = params.B * params.S * params.C;
    if (gid >= total) {
        return;
    }

    uint b = gid / (params.S * params.C);
    uint rem = gid % (params.S * params.C);
    uint s = rem / params.C;
    uint c_out = rem % params.C;

    uint x_base = b * params.S * params.C;
    uint y_index = x_base + s * params.C + c_out;

    float acc = params.has_bias ? bias[c_out] : 0.0f;

    for (uint k = 0; k < params.K; ++k) {
        int t_src = int(s) - int(k);
        if (t_src >= 0) {
            uint x_row = x_base + uint(t_src) * params.C;
            uint w_row = c_out * params.C * params.K + k * params.C;
            for (uint c_in = 0; c_in < params.C; ++c_in) {
                acc += x[x_row + c_in] * weight[w_row + c_in];
            }
        }
    }

    output[y_index] = acc;
}
