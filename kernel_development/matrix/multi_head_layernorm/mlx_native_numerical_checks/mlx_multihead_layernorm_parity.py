"""
MLX: Multi-Head LayerNorm parity check

Implements a simple head-aware layernorm and compares statistics to MLX GroupNorm
used in-place today. Validates per-head normalization behavior and stability.
"""

import mlx.core as mx
import mlx.nn as nn


def mh_layernorm(x, num_heads, eps=1e-6, force_float32=True):
    """Multi-head LayerNorm over DH per head, then flatten back.

    x: (B, NH*DH)
    returns: (B, NH*DH)
    """
    B, F = x.shape
    assert F % num_heads == 0
    DH = F // num_heads
    xh = x.reshape(B, num_heads, DH)
    in_dtype = xh.dtype
    if force_float32:
        xh = xh.astype(mx.float32)
    mean = mx.mean(xh, axis=-1, keepdims=True)
    var = mx.var(xh, axis=-1, keepdims=True)
    xh = (xh - mean) * mx.rsqrt(var + eps)
    xh = xh.astype(in_dtype)
    return xh.reshape(B, F)


def groupnorm_like(x, num_heads, eps=1e-6):
    """Approximate the current usage: GroupNorm(num_groups=NH, num_channels=NH*DH)
    We emulate it with NN GroupNorm over (N,L,C) by adding a length dim.
    """
    B, F = x.shape
    gn = nn.GroupNorm(num_heads, F)
    # MLX Conv/Norm expect (N,L,C); add L=1
    y = gn(x.reshape(B, 1, F))
    return y.reshape(B, F)


if __name__ == "__main__":
    mx.random.seed(0)
    B, NH, DH = 32, 8, 96
    x = mx.random.normal((B, NH * DH))
    y_mh = mh_layernorm(x, NH)
    y_gn = groupnorm_like(x, NH)
    # Compare per-head stats
    xh = x.reshape(B, NH, DH)
    yh = y_mh.reshape(B, NH, DH)
    mean = mx.mean(yh, axis=-1)
    var = mx.var(yh, axis=-1)
    print("MultiHead LN stats (mean,var over DH):")
    print(f"  mean≈0: {float(mx.mean(mx.abs(mean))):.3e}")
    print(f"  var≈1:  {float(mx.mean(mx.abs(var - 1))):.3e}")
    # Similarity to GroupNorm baseline
    diff = float(mx.max(mx.abs(y_mh - y_gn)))
    print(f"Max |Δ| vs GroupNorm: {diff:.3e}")
