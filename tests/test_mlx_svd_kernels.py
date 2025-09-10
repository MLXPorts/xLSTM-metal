
import mlx.core as mx
from mlx_src.mlx_fast_kernels.svd_kernels import power_iter_step_tiled


def test_svd_zstep_tiled():
    """Tests the tiled Z-step of the SVD power iteration."""
    mx.random.seed(2)
    m, n, k = 64, 48, 8
    A = mx.random.normal(shape=(m, n), dtype=mx.float32)
    V = mx.random.normal(shape=(n, k), dtype=mx.float32)
    Z_ref = mx.matmul(mx.transpose(A), mx.matmul(A, V))
    Z = power_iter_step_tiled(A, V)
    mx.eval(Z_ref, Z)
    assert float(mx.max(mx.abs(Z_ref - Z))) <= 1e-4
