// Deterministic 3-tap depthwise 1D convolution
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint batch;
    uint channels;
    uint length;
};

kernel void depthwise3(
    constant Params& p                 [[ buffer(0) ]],
    const device float* x              [[ buffer(1) ]], // shape [B,C,L] packed row-major
    const device float* k              [[ buffer(2) ]], // shape [C,3]
    device float* y                    [[ buffer(3) ]], // shape [B,C,L]
    uint gid                           [[ thread_position_in_grid ]]) {
    uint total = p.batch * p.channels * p.length;
    if (gid >= total) return;
    uint L = p.length;
    uint C = p.channels;
    uint b = gid / (C * L);
    uint r = gid % (C * L);
    uint c = r / L;
    uint t = r % L;
    // zero-pad by 2 on both sides, take window [t, t+1, t+2]
    float x0 = (t + 0 < L) ? x[(b*C + c)*L + (t + 0)] : 0.0f;
    float x1 = (t + 1 < L) ? x[(b*C + c)*L + (t + 1)] : 0.0f;
    float x2 = (t + 2 < L) ? x[(b*C + c)*L + (t + 2)] : 0.0f;
    // fixed order multiply-adds
    float w0 = k[c*3 + 0];
    float w1 = k[c*3 + 1];
    float w2 = k[c*3 + 2];
    float acc = (x0 * w0) + (x1 * w1);
    acc = acc + (x2 * w2);
    y[(b*C + c)*L + t] = acc;
}

