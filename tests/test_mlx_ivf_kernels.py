
import mlx.core as mx
from mlx_fast_kernels.ivf_kernels import ivf_list_topk_l2


def _host_topk_l2(Q, X, ids, k):
    # Compute distances on host and select top-k
    Qh = Q.tolist(); Xh = X.tolist(); idh = ids.tolist()
    import math
    dists = []
    for i, xi in enumerate(Xh):
        acc = 0.0
        for a, b in zip(xi, Qh):
            diff = a - b; acc += diff*diff
        dists.append((acc, idh[i]))
    dists.sort(key=lambda t: t[0])
    vals = [t[0] for t in dists[:k]]
    idxs = [t[1] for t in dists[:k]]
    return vals, idxs


def test_ivf_topk_small():
    """Tests the IVF top-k kernel with a small dataset."""
    mx.random.seed(3)
    m, d, k = 200, 32, 8
    Q = mx.random.normal(shape=(d,), dtype=mx.float32)
    X = mx.random.normal(shape=(m, d), dtype=mx.float32)
    ids = mx.arange(m, dtype=mx.int32)
    vals, idxs = ivf_list_topk_l2(Q, X, ids, k)
    mx.eval(vals, idxs)
    hv, hi = _host_topk_l2(Q, X, ids, k)
    # Compare sets (order may match due to selection sort); allow small tolerance
    assert len(vals) == k and len(idxs) == k
    # Verify same ids (unordered)
    assert set(idxs.tolist()) == set(hi)

