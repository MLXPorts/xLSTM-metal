import mlx.core as mx
import numpy as np
from xlstm_metal.blocks.mlstm_mlx.kernel import mlstm_sequential, mlstm_chunkwise

def test_recurrent_kernel():
    B, NH, S, QK_DH, V_DH = 1, 1, 128, 16, 16
    L = 64
    NC = S // L

    q = mx.random.normal((B, NH, S, QK_DH))
    k = mx.random.normal((B, NH, S, QK_DH))
    v = mx.random.normal((B, NH, S, V_DH))
    i_preact = mx.random.normal((B, NH, S))
    f_preact = mx.random.normal((B, NH, S))

    # Run sequential implementation to get expected states
    _, (c_expected, n_expected, m_expected) = mlstm_sequential(
        q, k, v, i_preact, f_preact, return_last_states=True
    )

    # Run chunkwise implementation
    _, (c_actual, n_actual, m_actual) = mlstm_chunkwise(
        q, k, v, i_preact, f_preact, chunk_size=L, return_last_states=True
    )

    # Compare states
    c_match = np.allclose(np.array(c_expected), np.array(c_actual), atol=1e-5)
    n_match = np.allclose(np.array(n_expected), np.array(n_actual), atol=1e-5)
    m_match = np.allclose(np.array(m_expected), np.array(m_actual), atol=1e-5)

    print(f"c_state match: {c_match}")
    print(f"n_state match: {n_match}")
    print(f"m_state match: {m_match}")

if __name__ == "__main__":
    test_recurrent_kernel()
