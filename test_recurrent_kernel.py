import mlx.core as mx
from xlstm_metal.blocks.mlstm_metal.bw_kernel_recurrent import mlstm_chunkwise_recurrent_bw_dC_metal
import numpy as np

def test_kernel():
    B = 1
    NH = 1
    S = 32
    DHQK = 16
    DHHV = 16
    NC = 2
    L = 16
    qk_scale = 1.0
    save_states_every_nth_chunk = 1

    matQ = mx.random.normal((B, NH, S, DHQK))
    vecF = mx.random.normal((B, NH, S))
    scaM_inter = mx.random.normal((B, NH, NC + 1))
    vecM_combine = mx.random.normal((B, NH, S))
    matDeltaH = mx.random.normal((B, NH, S, DHHV))
    vecN_out = mx.random.normal((B, NH, S))
    matDeltaC_last = mx.random.normal((B, NH, DHQK, DHHV))

    output = mlstm_chunkwise_recurrent_bw_dC_metal(
        matQ,
        vecF,
        scaM_inter,
        vecM_combine,
        matDeltaH,
        vecN_out,
        matDeltaC_last,
        NC=NC,
        L=L,
        qk_scale=qk_scale,
        save_states_every_nth_chunk=save_states_every_nth_chunk
    )

    print("Output shape:", output.shape)
    print("Output sum:", mx.sum(output).item())

    # The kernel should modify the output buffer, so it shouldn't be all zeros.
    if mx.sum(output).item() == 0.0:
        print("Test failed: Output is all zeros.")
    else:
        print("Test passed: Output contains non-zero values.")

if __name__ == "__main__":
    test_kernel()

