"""
Bench and parity check: Multi-Head LayerNorm — MLX ops vs Metal simd kernel.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

# Ensure we can import kernels from this lab package despite hyphen in folder name
_HERE = Path(__file__).resolve().parent
_KER = _HERE / "kernels"
if str(_KER) not in sys.path:
    sys.path.insert(0, str(_KER))
from mhln_kernels import mh_layernorm_simd  # type: ignore


def mh_layernorm_mlx(x, num_heads, eps=1e-6):
    """

    :param x:
    :param num_heads:
    :param eps:
    :return:
    """
    B, F = x.shape
    assert F % num_heads == 0
    DH = F // num_heads
    xh = x.reshape(B, num_heads, DH)
    in_dtype = xh.dtype
    xf = xh.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.var(xf, axis=-1, keepdims=True)
    y = ((xf - mean) * mx.rsqrt(var + eps)).astype(in_dtype)
    return y.reshape(B, F)


def run_once(B=64, NH=8, DH=128):
    """

    :param B:
    :param NH:
    :param DH:
    :return:
    """
    mx.random.seed(0)
    x = mx.random.normal((B, NH * DH))
    # MLX ops
    t0 = time.time()
    y_mlx = mh_layernorm_mlx(x, NH)
    mx.eval(y_mlx)
    t_mlx = time.time() - t0
    # Kernel
    X3 = x.reshape(B, NH, DH)
    t1 = time.time()
    y_k = mh_layernorm_simd(X3)
    mx.eval(y_k)
    t_k = time.time() - t1
    # Parity
    diff = float(mx.max(mx.abs(y_k.reshape(B, NH * DH) - y_mlx)))
    return diff, t_mlx, t_k


if __name__ == "__main__":
    for DH in (64, 96, 128, 192):
        diff, t_mlx, t_k = run_once(DH=DH)
        print(f"B=64 NH=8 DH={DH:3d}  max|Δ|={diff:.3e}  mlx={t_mlx * 1e3:.2f} ms  kernel={t_k * 1e3:.2f} ms")
