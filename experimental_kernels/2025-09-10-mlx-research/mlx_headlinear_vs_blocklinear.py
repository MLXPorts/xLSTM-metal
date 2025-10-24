"""
MLX: HeadLinear vs Block-Diagonal Linear

Compares correctness and rough cost of two strategies:
- Block-diagonal: build a full block-diagonal weight then do one large matmul.
- Per-head linear: split features per head and apply per-head weight directly.

Notes
- This isolates the cost of constructing the block-diagonal matrix.
- The per-head path uses H small GEMMs; depending on MLX batching, this may
  be slower for tiny heads but avoids the (H·D)^2 block matrix and its write.
"""

import time
import math
import mlx.core as mx


def block_diag(weights):
    """Construct a block-diagonal (sum of shapes along diag)."""
    rows = sum(w.shape[0] for w in weights)
    cols = sum(w.shape[1] for w in weights)
    out = mx.zeros((rows, cols), dtype=weights[0].dtype)
    r = 0
    c = 0
    for w in weights:
        rr, cc = w.shape
        out[r:r+rr, c:c+cc] = w
        r += rr
        c += cc
    return out


def head_split_linear(x, W_list):
    """Apply per-head linear: concatenate [x_h @ W_h.T].

    x: (B, H*Di)
    W_list: list of H arrays (Do, Di)
    Returns: (B, H*Do)
    """
    B, F = x.shape
    H = len(W_list)
    Di = F // H
    outs = []
    for h in range(H):
        xh = x[:, h*Di:(h+1)*Di]
        Wh = W_list[h]
        outs.append(xh @ mx.transpose(Wh))
    return mx.concatenate(outs, axis=-1)


def run_once(B=32, H=8, Di=64, Do=64, dtype=mx.float32):
    mx.random.seed(0)
    x = mx.random.normal((B, H*Di), dtype=dtype)
    W_list = [mx.random.normal((Do, Di), dtype=dtype) for _ in range(H)]

    # Correctness
    W_block = block_diag(W_list)
    y_block = x @ mx.transpose(W_block)
    y_head = head_split_linear(x, W_list)
    diff = float(mx.max(mx.abs(y_block - y_head)))

    # Cost: block-diag construction
    t0 = time.time()
    for _ in range(10):
        Wb = block_diag(W_list)
        mx.eval(Wb)
    t_block_build = (time.time() - t0) / 10

    # Cost: one big matmul vs H small matmuls
    t1 = time.time()
    for _ in range(10):
        yb = x @ mx.transpose(W_block)
        mx.eval(yb)
    t_big = (time.time() - t1) / 10

    t2 = time.time()
    for _ in range(10):
        yh = head_split_linear(x, W_list)
        mx.eval(yh)
    t_heads = (time.time() - t2) / 10

    return diff, t_block_build, t_big, t_heads


if __name__ == "__main__":
    configs = [
        (32, 8, 64, 64),
        (32, 8, 96, 96),
        (16, 16, 64, 64),
    ]
    print("HeadLinear vs Block-Diagonal (MLX)")
    for B, H, Di, Do in configs:
        diff, t_build, t_big, t_heads = run_once(B, H, Di, Do)
        print(f"B={B} H={H} Di={Di} Do={Do}  max|Δ|={diff:.3e}")
        print(f"  block-diag build: {t_build*1e3:.2f} ms")
        print(f"  big GEMM       : {t_big*1e3:.2f} ms")
        print(f"  per-head GEMMs : {t_heads*1e3:.2f} ms")

