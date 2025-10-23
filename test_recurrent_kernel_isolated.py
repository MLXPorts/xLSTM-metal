import mlx.core as mx
import numpy as np
from xlstm_metal.blocks.mlstm_metal.fw_kernel_recurrent import mlstm_chunkwise_recurrent_fw_C_metal
from xlstm_metal.blocks.mlstm_mlx.kernel import mlstm_recurrent_step

def sequential_recurrent(q, k, v, f_preact, i_preact, c_initial, n_initial, m_initial, NC, L):
    c_state = mx.array(c_initial)
    n_state = mx.array(n_initial)
    m_state = mx.array(m_initial)

    for i in range(NC):
        q_chunk = q[:, :, i*L:(i+1)*L, :]
        k_chunk = k[:, :, i*L:(i+1)*L, :]
        v_chunk = v[:, :, i*L:(i+1)*L, :]
        f_chunk = f_preact[:, :, i*L:(i+1)*L]
        i_chunk = i_preact[:, :, i*L:(i+1)*L]

        _, c_state, n_state, m_state = mlstm_recurrent_step(
            q_chunk[:, :, 0, :], k_chunk[:, :, 0, :], v_chunk[:, :, 0, :], i_chunk[:, :, 0], f_chunk[:, :, 0], c_state, n_state, m_state
        )

    return c_state, n_state, m_state

def test_recurrent_kernel_isolated():
    B, NH, S, QK_DH, V_DH = 1, 1, 128, 16, 16
    L = 64
    NC = S // L

    q = mx.random.normal((B, NH, S, QK_DH))
    k = mx.random.normal((B, NH, S, QK_DH))
    v = mx.random.normal((B, NH, S, V_DH))
    i_preact = mx.random.normal((B, NH, S))
    f_preact = mx.random.normal((B, NH, S))
    c_initial = mx.random.normal((B, NH, QK_DH, V_DH))
    n_initial = mx.random.normal((B, NH, QK_DH))
    m_initial = mx.random.normal((B, NH))

    # Python reference calculation
    c_expected, n_expected, m_expected = sequential_recurrent(
        q, k, v, f_preact, i_preact, c_initial, n_initial, m_initial, NC, L
    )

    # Metal kernel calculation
    dbg_buffer = mx.zeros((L * 3 + 1,), dtype=mx.float32)
    c_states_actual, n_states_actual, m_states_actual, dbg_out = mlstm_chunkwise_recurrent_fw_C_metal(
        matK=k, matV=v, vecF=f_preact, vecI=i_preact,
        matC_initial=c_initial, vecN_initial=n_initial, scaMinter_initial=m_initial,
        NC=NC, L=L, dbg=dbg_buffer
    )
    c_actual = c_states_actual[:, :, -QK_DH:, :].reshape(B, NH, QK_DH, V_DH)
    n_actual = n_states_actual[:, :, -QK_DH:].reshape(B, NH, QK_DH)
    m_actual = m_states_actual[:, :, -1]

    # --- Debugging vecA and scaG ---
    f_chunk_last = f_preact[:, :, -L:]
    i_chunk_last = i_preact[:, :, -L:]

    # Canonical A and G (see Transformers modeling_xlstm):
    # vecB = cumsum(logsigmoid(F)); vecA = vecB_last - vecB + vecI; scaG = vecB_last
    f_log_sigmoid = -mx.log(1.0 + mx.exp(-f_chunk_last))
    vecB = mx.cumsum(f_log_sigmoid, axis=-1)
    vecA_expected = vecB[:, :, -1, None] - vecB + i_chunk_last
    scaG_expected = vecB[:, :, -1]

    print("vecA from dbg:", dbg_out[0:L])
    print("vecA expected:", vecA_expected[0, 0, :])
    print("vecFlogsig_masked from dbg:", dbg_out[L:2*L])
    print("(dbg shows masked (shifted) logsig for inspection only)")
    print("vecI from dbg:", dbg_out[2*L:3*L])
    print("vecI expected:", i_chunk_last[0, 0, :])
    print("scaG from dbg:", dbg_out[3*L])
    print("scaG expected:", scaG_expected[0, 0])

    # --- Comparison ---
    print("\n--- State Comparison ---")
    c_match = np.allclose(np.array(c_expected), np.array(c_actual), atol=1e-5)
    n_match = np.allclose(np.array(n_expected), np.array(n_actual), atol=1e-5)
    m_match = np.allclose(np.array(m_expected), np.array(m_actual), atol=1e-5)

    print(f"c_state match: {c_match}")
    print(f"n_state match: {n_match}")
    print(f"m_state match: {m_match}")

if __name__ == "__main__":
    test_recurrent_kernel_isolated()
