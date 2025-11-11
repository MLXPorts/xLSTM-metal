// Minimal complex multiply with deterministic accumulation order
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n; // number of complex elements
};

kernel void complex_mul(
    constant Params& p           [[ buffer(0) ]],
    const device float2* a       [[ buffer(1) ]],
    const device float2* b       [[ buffer(2) ]],
    device float2* out           [[ buffer(3) ]],
    uint gid                     [[ thread_position_in_grid ]]) {
    if (gid >= p.n) return;
    float ar = a[gid].x;
    float ai = a[gid].y;
    float br = b[gid].x;
    float bi = b[gid].y;
    // Enforce a fixed evaluation order (no FMA)
    float real = (ar * br) - (ai * bi);
    float imag = (ar * bi) + (ai * br);
    out[gid] = float2(real, imag);
}

