#!/usr/bin/env python
"""
Metal Kernel Dispatch Layer for MLX

Pure kernel-based operations - Python only manages buffers and chains kernel calls.
No Python loops, no scalar operations - everything happens in Metal.

Architecture:
    Python: mx.array buffer management only
    Metal: All computation (depthwise conv, complex mul, FFT)

Usage:
    # Bad (old way):
    for i in range(channels):
        out[i] = conv(x[i], kernel[i])

    # Good (kernel way):
    out = depthwise_conv_kernel(x, kernel)

Note:
    MLX JIT generates kernel function signatures automatically from input_names/output_names.
    Kernel source should contain ONLY the body - no function signature.
"""

import mlx.core as mx


_HEADER = """#include <metal_stdlib>
using namespace metal;
"""


# ============================================================================
# Depthwise Convolution Kernel
# ============================================================================

_DEPTHWISE_KERNEL = None

_DEPTHWISE_SOURCE = r"""
    // Extract dimensions from params
    uint B = params[0];
    uint C = params[1];
    uint L = params[2];

    // Global thread ID
    uint tid = thread_position_in_grid.x;
    uint total = B * C * L;
    if (tid >= total) return;

    // Decode indices: tid = b * (C * L) + c * L + t
    uint b = tid / (C * L);
    uint rem = tid % (C * L);
    uint c = rem / L;
    uint t = rem % L;

    // Zero-pad input: access x[b, c, t] with bounds checking
    float v0 = (t >= 1) ? x[b * C * L + c * L + (t - 1)] : 0.0f;
    float v1 = x[b * C * L + c * L + t];
    float v2 = (t + 1 < L) ? x[b * C * L + c * L + (t + 1)] : 0.0f;

    // Load kernel weights for this channel
    float k0 = k[c * 3 + 0];
    float k1 = k[c * 3 + 1];
    float k2 = k[c * 3 + 2];

    // Convolve: fixed order to ensure determinism
    float acc = v0 * k0;
    acc = fma(v1, k1, acc);
    acc = fma(v2, k2, acc);

    // Write output
    y[tid] = acc;
"""


def depthwise_conv_3tap(x, kernel):
    """
    Depthwise 3-tap convolution using Metal kernel

    Args:
        x: (batch, channels, length) - input signal
        kernel: (channels, 3) - per-channel 3-tap filters

    Returns:
        y: (batch, channels, length) - convolved output
    """
    global _DEPTHWISE_KERNEL

    batch, channels, length = x.shape

    # Ensure float32
    x = x.astype(mx.float32)
    kernel = kernel.astype(mx.float32)

    # Compile kernel on first use
    if _DEPTHWISE_KERNEL is None:
        _DEPTHWISE_KERNEL = mx.fast.metal_kernel(
            name="depthwise3",
            input_names=["params", "x", "k"],
            output_names=["y"],
            header=_HEADER,
            source=_DEPTHWISE_SOURCE
        )

    # Prepare params buffer
    params = mx.array([batch, channels, length], dtype=mx.uint32)

    # Configure grid and threadgroup using MLX scalars in tuples
    batch_mx = mx.array(batch, dtype=mx.uint32)
    channels_mx = mx.array(channels, dtype=mx.uint32)
    length_mx = mx.array(length, dtype=mx.uint32)
    total_threads = mx.multiply(mx.multiply(batch_mx, channels_mx), length_mx)
    tpg = mx.array(256, dtype=mx.uint32)
    one = mx.array(1, dtype=mx.uint32)
    num_groups = mx.divide(mx.add(total_threads, mx.subtract(tpg, one)), tpg)
    grid = (num_groups, one, one)
    threadgroup = (tpg, one, one)

    # Dispatch kernel
    (y,) = _DEPTHWISE_KERNEL(
        inputs=[params, x, kernel],
        output_shapes=[(batch, channels, length)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup
    )

    return y


# ============================================================================
# Complex Multiply Kernel
# ============================================================================

_COMPLEX_MUL_KERNEL = None

_COMPLEX_MUL_SOURCE = r"""
    // Get total elements from params
    uint n = params[0];
    uint tid = thread_position_in_grid.x;
    if (tid >= n) return;

    // Manual complex multiply with fixed evaluation order for determinism
    // (a_r + i*a_i) * (b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
    complex64_t a_val = a[tid];
    complex64_t b_val = b[tid];

    float a_r = a_val.real;
    float a_i = a_val.imag;
    float b_r = b_val.real;
    float b_i = b_val.imag;

    // Fixed order: no FMA, explicit parentheses
    float real_part = (a_r * b_r) - (a_i * b_i);
    float imag_part = (a_r * b_i) + (a_i * b_r);

    out[tid] = complex64_t(real_part, imag_part);
"""


def complex_multiply(a, b):
    """
    Complex multiplication using Metal kernel

    Args:
        a: (..., freq_bins) complex64 - first operand
        b: (..., freq_bins) complex64 - second operand

    Returns:
        out: (..., freq_bins) complex64 - a * b
    """
    global _COMPLEX_MUL_KERNEL

    # Get shape
    shape = a.shape

    # Flatten to 1D for kernel
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    total_elements = a_flat.shape[0]

    # Compile kernel on first use
    if _COMPLEX_MUL_KERNEL is None:
        _COMPLEX_MUL_KERNEL = mx.fast.metal_kernel(
            name="complex_mul",
            input_names=["params", "a", "b"],
            output_names=["out"],
            header=_HEADER,
            source=_COMPLEX_MUL_SOURCE
        )

    # Prepare params
    total_elements_mx = mx.array(total_elements, dtype=mx.uint32)
    params = mx.array([total_elements], dtype=mx.uint32)

    # Configure grid and threadgroup using MLX scalars in tuples
    tpg = mx.array(256, dtype=mx.uint32)
    one = mx.array(1, dtype=mx.uint32)
    num_groups = mx.divide(mx.add(total_elements_mx, mx.subtract(tpg, one)), tpg)
    grid = (num_groups, one, one)
    threadgroup = (tpg, one, one)

    # Dispatch kernel
    (out_flat,) = _COMPLEX_MUL_KERNEL(
        inputs=[params, a_flat, b_flat],
        output_shapes=[(total_elements,)],
        output_dtypes=[mx.complex64],
        grid=grid,
        threadgroup=threadgroup
    )

    # Reshape back
    out = out_flat.reshape(shape)
    return out


# ============================================================================
# Kernel Registry
# ============================================================================

AVAILABLE_KERNELS = {
    'depthwise_conv_3tap': depthwise_conv_3tap,
    'complex_multiply': complex_multiply,
}

# ============================================================================
# Extended-Precision (Double-Double) Complex Multiply
# ============================================================================

_COMPLEX_MUL_DD_KERNEL = None

_DD_HEADER = _HEADER + r"""
// Double-Double support (minimal subset)
struct dd_t { float hi; float lo; };
inline dd_t quick_two_sum(float a, float b) { float s = a + b; float e = b - (s - a); return dd_t{s, e}; }
inline dd_t two_sum(float a, float b) { float s = a + b; float v = s - a; float e = (a - (s - v)) + (b - v); 
return dd_t{s, e}; }
inline dd_t two_prod(float a, float b) { float p = a * b; float e = fma(a, b, -p); return dd_t{p, e}; }
inline dd_t dd_add(dd_t a, dd_t b) { dd_t s = two_sum(a.hi, b.hi); dd_t t = two_sum(a.lo, b.lo); s.lo += t.hi; 
s = quick_two_sum(s.hi, s.lo); s.lo += t.lo; s = quick_two_sum(s.hi, s.lo); return s; }
inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_t{-b.hi, -b.lo}); }
inline dd_t dd_mul(dd_t a, dd_t b) { dd_t p = two_prod(a.hi, b.hi); p.lo += a.hi * b.lo + a.lo * b.hi; 
p = quick_two_sum(p.hi, p.lo); return p; }
inline float dd_to_float(dd_t a) { return a.hi + a.lo; }

struct cdd_t { dd_t re; dd_t im; };
inline cdd_t cdd_mul(cdd_t a, cdd_t b) { dd_t ac = dd_mul(a.re, b.re); dd_t bd = dd_mul(a.im, b.im); 
dd_t re = dd_sub(ac, bd); dd_t ad = dd_mul(a.re, b.im); dd_t bc = dd_mul(a.im, b.re); dd_t im = dd_add(ad, bc); return cdd_t{re, im}; }
"""

_COMPLEX_MUL_DD_SOURCE = r"""
    // Get total elements from params
    uint n = params[0];
    uint tid = thread_position_in_grid.x;
    if (tid >= n) return;

    // Load complex64 inputs
    complex64_t av = a[tid];
    complex64_t bv = b[tid];

    // Lift to DD
    cdd_t aa = cdd_t{ dd_t{av.real, 0.0f}, dd_t{av.imag, 0.0f} };
    cdd_t bb = cdd_t{ dd_t{bv.real, 0.0f}, dd_t{bv.imag, 0.0f} };

    // Multiply in extended precision
    cdd_t rr = cdd_mul(aa, bb);

    // Single rounding to float32
    float real_part = dd_to_float(rr.re);
    float imag_part = dd_to_float(rr.im);
    out[tid] = complex64_t(real_part, imag_part);
"""

def complex_multiply_dd(a, b):
    """
    Extended-precision complex multiply using double-double arithmetic.

    Args:
        a, b: arrays with dtype=mx.complex64, arbitrary shape

    Returns:
        out: mx.complex64 array, same shape, computed via DD multiply then rounded once.
    """
    global _COMPLEX_MUL_DD_KERNEL

    shape = a.shape
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    n = a_flat.shape[0]

    if _COMPLEX_MUL_DD_KERNEL is None:
        _COMPLEX_MUL_DD_KERNEL = mx.fast.metal_kernel(
            name="complex_mul_dd",
            input_names=["params", "a", "b"],
            output_names=["out"],
            header=_DD_HEADER,
            source=_COMPLEX_MUL_DD_SOURCE,
        )

    n_mx = mx.array(n, dtype=mx.uint32)
    params = mx.array([n], dtype=mx.uint32)
    tpg = mx.array(256, dtype=mx.uint32)
    one = mx.array(1, dtype=mx.uint32)
    num_groups = mx.divide(mx.add(n_mx, mx.subtract(tpg, one)), tpg)
    grid = (num_groups, one, one)
    threadgroup = (tpg, one, one)

    (out_flat,) = _COMPLEX_MUL_DD_KERNEL(
        inputs=[params, a_flat, b_flat],
        output_shapes=[(n,)],
        output_dtypes=[mx.complex64],
        grid=grid,
        threadgroup=threadgroup,
    )

    return out_flat.reshape(shape)

AVAILABLE_KERNELS["complex_multiply_dd"] = complex_multiply_dd
