"""
MLX Soft-Cap: Pure MLX vs Metal Kernel

Validates numeric parity and measures rough throughput for applying
soft_cap(x, cap) both via pure MLX ops and via a small Metal kernel
in mlx_fast_kernels/shaders.py.
"""

import time

import mlx.core as mx

from kernel_development.soft_cap.mlx_fast_kernels.softcap import SoftCapMLXFastKernel

metal_soft_cap = SoftCapMLXFastKernel()


def soft_cap_mlx(x, cap):
    """

    :param x:
    :param cap:
    :return:
    """
    return cap * mx.tanh(x / cap)


def bench_once(n=1_000_000, cap=15.0, dtype=mx.float32):
    """

    :param n:
    :param cap:
    :param dtype:
    :return:
    """
    mx.random.seed(0)
    x = mx.random.normal((n,), dtype=dtype)

    # Numeric parity
    y_mlx = soft_cap_mlx(x, cap)
    y_mtl = metal_soft_cap(x, cap)
    diff = float(mx.max(mx.abs(y_mlx - y_mtl)))

    # Timing
    iters = 20
    t0 = time.time()
    for _ in range(iters):
        ym = soft_cap_mlx(x, cap)
        mx.eval(ym)
    t_mlx = (time.time() - t0) / iters

    t1 = time.time()
    for _ in range(iters):
        yk = metal_soft_cap(x, cap)
        mx.eval(yk)
    t_mtl = (time.time() - t1) / iters

    return diff, t_mlx, t_mtl


if __name__ == "__main__":
    for n in (1_000_0, 1_000_00, 1_000_000):
        diff, t_mlx, t_mtl = bench_once(n=n)
        print(f"n={n:8d}  max|Δ|={diff:.3e}  mlx={t_mlx * 1e3:.2f} ms  metal={t_mtl * 1e3:.2f} ms")
