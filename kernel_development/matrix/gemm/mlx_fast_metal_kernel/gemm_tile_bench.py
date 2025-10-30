"""
GEMM Tile Bench (MLX Metal)

Benchmarks gemm_av (A@V) and gemm_at_b (A.T@B) with different tile shapes.
Includes variants discussed in WWDC-inspired notes (16x16 default; 32x8; 8x32; 16x8; 8x16).
"""

import time

import mlx.core as mx
from gemm_kernels import gemm_av, gemm_at_b, set_gemm_tiles


def bench_gemm(m=1024, n=1024, k=1024, tiles=("16x16", "16x16")):
    """

    :param m:
    :param n:
    :param k:
    :param tiles:
    :return:
    """
    mx.random.seed(0)
    A = mx.random.normal((m, n))
    V = mx.random.normal((n, k))
    B = mx.random.normal((m, k))

    # Warm
    _ = gemm_av(A, V); mx.eval(_)
    _ = gemm_at_b(A, B); mx.eval(_)

    av, atb = tiles
    set_gemm_tiles(av, atb)

    iters = 10
    t0 = time.time()
    for _ in range(iters):
        C = gemm_av(A, V); mx.eval(C)
    t_av = (time.time() - t0) / iters

    t1 = time.time()
    for _ in range(iters):
        Z = gemm_at_b(A, B); mx.eval(Z)
    t_atb = (time.time() - t1) / iters

    return t_av, t_atb


if __name__ == "__main__":
    shapes = [(1024, 1024, 1024)]
    tiles = [
        ("16x16", "16x16"),
        ("32x8", "8x32"),
        ("8x32", "32x8"),
        ("16x8", "8x16"),  # hpc16x8 idea
        ("8x16", "16x8"),
    ]
    for (m, n, k) in shapes:
        print(f"m={m} n={n} k={k}")
        for av, atb in tiles:
            t_av, t_atb = bench_gemm(m, n, k, tiles=(av, atb))
            print(f"  tiles av={av:>6} atb={atb:>6}:  A@V={t_av*1e3:.2f} ms,  A.T@B={t_atb*1e3:.2f} ms")
