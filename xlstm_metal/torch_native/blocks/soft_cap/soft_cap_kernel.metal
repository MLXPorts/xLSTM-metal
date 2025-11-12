#include <metal_stdlib>
using namespace metal;

kernel void soft_cap_kernel(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant float& cap_value       [[buffer(2)]],
    constant uint& numel            [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]) {
    if (gid >= numel) {
        return;
    }
    float x = input[gid];
    output[gid] = cap_value * tanh(x / cap_value);
}
