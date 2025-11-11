"""Metal depthwise causal conv kernel (per-feature)."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_SRC = r"""
// params: [B, S, C, K]
uint B = params[0];
uint S = params[1];
uint C = params[2];
uint K = params[3];

uint tid = thread_position_in_grid.x;
uint total = B * S * C;
if (tid >= total) return;

uint b = tid / (S * C);
uint rem = tid % (S * C);
uint s = rem / C;
uint c = rem % C;

uint x_base = b * S * C;
uint y_index = x_base + s * C + c;

float acc = has_bias ? bias[c] : 0.0f;

for (uint k = 0; k < K; ++k) {
    int t_src = int(s) - int(k);
    if (t_src >= 0) {
        uint x_index = x_base + uint(t_src) * C + c;
        uint w_index = c * K + k;
        acc += x[x_index] * weight[w_index];
    }
}

output[y_index] = acc;
"""

_KERNEL = None


def _compile_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="dw_causal_conv",
            input_names=["params", "x", "weight", "bias", "has_bias"],
            output_names=["output"],
            header=_HEADER,
            source=_SRC,
        )
    return _KERNEL


def metal_causal_conv1d_depthwise(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
) -> mx.array:
    """Depthwise causal conv (groups=channels)."""
    B, S, C = x.shape
    Cw, K = weight.shape
    assert Cw == C, "weight must be [C, K] for depthwise conv"

    params = mx.array([B, S, C, K], dtype=mx.uint32)
    has_bias = mx.array(1 if bias is not None else 0, dtype=mx.uint32)
    if bias is None:
        bias = mx.zeros((C,), dtype=x.dtype)

    total = B * S * C
    grid = (int(total), 1, 1)
    threadgroup = (min(int(total), 256), 1, 1)

    kernel = _compile_kernel()
    y, = kernel(
        inputs=[
            params,
            x.astype(mx.float32),
            weight.reshape(-1).astype(mx.float32),
            bias.astype(mx.float32),
            has_bias,
        ],
        output_shapes=[(B * S * C,)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )
    return y.reshape(B, S, C).astype(x.dtype)


__all__ = ["metal_causal_conv1d_depthwise"]
