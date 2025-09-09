
import os
import mlx.core as mx
from mlx_fast_kernels.gemm_kernels import gemm_av, gemm_at_b, set_gemm_tiles


def _check_once(m, n, k, tm, t):
    mx.random.seed(0)
    A = mx.random.normal(shape=(m, n), dtype=mx.float32)
    V = mx.random.normal(shape=(n, k), dtype=mx.float32)
    set_gemm_tiles(av=(tm, t), atb=(t, t))
    B_ref = mx.matmul(A, V); B = gemm_av(A, V); mx.eval(B_ref, B)
    assert float(mx.max(mx.abs(B_ref - B))) <= 1e-4
    Z_ref = mx.matmul(mx.transpose(A), B_ref); Z = gemm_at_b(A, B); mx.eval(Z_ref, Z)
    assert float(mx.max(mx.abs(Z_ref - Z))) <= 1e-4


def test_gemm_parity_multi():
    shapes = [(64,128,32), (33,29,31), (128,96,80)]
    tiles = [(16,16), (32,8), (8,32)]
    for m,n,k in shapes:
        for tm,t in tiles:
            _check_once(m,n,k,tm,t)

