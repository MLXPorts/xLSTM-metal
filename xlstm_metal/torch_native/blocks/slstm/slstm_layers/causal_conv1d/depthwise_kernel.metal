#include <metal_stdlib>
using namespace metal;

struct DepthParams {
    uint B;
    uint S;
    uint C;
    uint K;
    uint has_bias;
};

kernel void causal_conv1d_depthwise(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    constant DepthParams& params [[buffer(3)]],
    device float* output         [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]) {
    uint total = params.B * params.S * params.C;
    if (gid >= total) {
        return;
    }

    uint b = gid / (params.S * params.C);
    uint rem = gid % (params.S * params.C);
    uint s = rem / params.C;
    uint c = rem % params.C;

    uint x_base = b * params.S * params.C;
    uint y_index = x_base + s * params.C + c;

    float acc = params.has_bias ? bias[c] : 0.0f;

    for (uint k = 0; k < params.K; ++k) {
        int t_src = int(s) - int(k);
        if (t_src >= 0) {
            uint x_index = x_base + uint(t_src) * params.C + c;
            uint w_index = c * params.K + k;
            acc += x[x_index] * weight[w_index];
        }
    }

    output[y_index] = acc;
}
