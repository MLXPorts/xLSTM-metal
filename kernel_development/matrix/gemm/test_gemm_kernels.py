
"""
Quick numeric checks for MLX GEMM kernels (naive vs tiled vs mx.matmul).

Run:
  PYTHONPATH=. python lab/test_gemm_kernels.py

Set XLSTM_GEMM_NAIVE=1 to force naive path.
"""
import os
import time

import mlx.core as mx
from .mlx_fast_kernels.gemm_kernels import gemm_av, gemm_at_b, set_gemm_tiles


def check(shape=(64, 128, 32), tiles=(16, 16)):
    m, n, k = shape
    set_gemm_tiles(av=tiles, atb=tiles)
    mx.random.seed(0)
    A = mx.random.normal(shape=(m, n), dtype=mx.float32)
    V = mx.random.normal(shape=(n, k), dtype=mx.float32)

    # AV
    t0 = time.time(); B0 = mx.matmul(A, V); mx.eval(B0); t0 = time.time() - t0
    t1 = time.time(); B1 = gemm_av(A, V); mx.eval(B1); t1 = time.time() - t1
    diff_av = float(mx.max(mx.abs(B0 - B1)))

    # AT_B
    t2 = time.time(); Z0 = mx.matmul(mx.transpose(A), B0); mx.eval(Z0); t2 = time.time() - t2
    t3 = time.time(); Z1 = gemm_at_b(A, B0); mx.eval(Z1); t3 = time.time() - t3
    diff_atb = float(mx.max(mx.abs(Z0 - Z1)))

    print(f"Tiles={tiles}, shape={shape} :: AV diff={diff_av:.3e} ({t1:.4f}s vs {t0:.4f}s), AT_B diff={diff_atb:.3e} ({t3:.4f}s vs {t2:.4f}s)")


def main():
    """The main function of the script."""
    naive = os.environ.get("XLSTM_GEMM_NAIVE", "0") == "1"
    print(f"Naive path: {naive}")
    # Odd shapes to stress partial tiles
    shapes = [(64,128,32), (33,29,31), (128,96,80)]
    tiles_list = [(16,16), (32,8), (8,32)]
    for shp in shapes:
        for tiles in tiles_list:
            check(shp, tiles)
    # Tiny case to catch indexing mistakes
    A = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    V = mx.arange(6, dtype=mx.float32).reshape(3, 2)
    B0 = mx.matmul(A, V)
    B1 = gemm_av(A, V)
    mx.eval(B0, B1)
    print("Tiny AV diff:", float(mx.max(mx.abs(B0 - B1))))


if __name__ == "__main__":
    main()
