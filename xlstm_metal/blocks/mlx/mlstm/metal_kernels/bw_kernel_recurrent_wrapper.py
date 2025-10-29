"""Python wrapper for backward recurrent Metal kernel."""

import mlx.core as mx
from typing import Tuple, Optional
import struct


def mlstm_chunkwise_recurrent_bw_dC_metal(
    matQ: mx.array,  # (B, NH, S, DHQK)
    vecF: mx.array,  # (B, NH, S)
    scaM_inter: mx.array,  # (B, NH, NC+1)
    vecM_combine: mx.array,  # (B, NH, S)
    matDeltaH: mx.array,  # (B, NH, S, DHHV)
    vecN_out: mx.array,  # (B, NH, S)
    matDeltaC_last: Optional[mx.array],  # (B, NH, DHQK, DHHV)
    NC: int,
    L: int,
    qk_scale: float,
    siz_b_DHQK: int = 16,
    siz_b_DHHV: int = 16,
    save_states_every_nth_chunk: int = 1,
    eps: float = 1e-6,
) -> mx.array:
    """
    Metal kernel for backward recurrent computation of mLSTM gradient deltas.

    Backpropagates through chunks in reverse order (last to first).

    Returns:
        matDeltaC_states: (B, NH, (NC+1)*DHQK, DHHV) - gradient deltas for C states
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matDeltaH.shape[3]

    # Prepare parameter buffer
    USE_LAST_STATE = 1 if matDeltaC_last is not None else 0

    # Pack floats as reinterpreted uint32
    qk_scale_bits = struct.unpack('I', struct.pack('f', qk_scale))[0]
    eps_bits = struct.unpack('I', struct.pack('f', eps))[0]

    params = mx.array([B, NH, S, DHQK, DHHV, NC, L, siz_b_DHQK, siz_b_DHHV,
                       save_states_every_nth_chunk, USE_LAST_STATE,
                       qk_scale_bits, eps_bits], dtype=mx.uint32)

    # Prepare strides buffer
    strides = mx.array([
        NH * S * DHQK,  # str_matQ_B_NH
        DHQK,           # str_matQ_S
        1,              # str_matQ_DHQK
        NH * S,         # str_vecF_B_NH
        NC + 1,         # str_scaM_inter_B_NH
        1,              # str_scaM_inter_NC
        NH * S,         # str_vecM_combine_B_NH
        1,              # str_vecM_combine_S
        NH * S * DHHV,  # str_matDeltaH_B_NH
        DHHV,           # str_matDeltaH_S
        1,              # str_matDeltaH_DHHV
        NH * S,         # str_vecN_out_B_NH
        1,              # str_vecN_out_S
        NH * DHQK * DHHV,  # str_matDeltaC_last_B_NH
        DHHV,           # str_matDeltaC_last_DHQK
        1,              # str_matDeltaC_last_DHHV
        (NC + 1) * DHQK * DHHV,  # str_matDeltaC_states_B_NH
        DHHV,           # str_matDeltaC_states_NCDHQK
        1,              # str_matDeltaC_states_DHHV
    ], dtype=mx.uint32)

    # Allocate output
    matDeltaC_states = mx.zeros((B, NH, (NC + 1) * DHQK, DHHV), dtype=matQ.dtype)

    # Default last state if not provided
    if matDeltaC_last is None:
        matDeltaC_last = mx.zeros((B, NH, DHQK, DHHV), dtype=matQ.dtype)

    # Import Metal kernel source from main file
    from bw_kernel_recurrent import _HEADER, _RECURRENT_BW_DC_SRC

    # Build kernel
    kernel = mx.fast.metal_kernel(
        name="mlstm_recurrent_bw_dC",
        input_names=["matQ", "vecF", "scaM_inter", "vecM_combine", "matDeltaH",
                     "vecN_out", "matDeltaC_last", "params", "strides"],
        output_names=["matDeltaC_states"],
        header=_HEADER,
        source=_RECURRENT_BW_DC_SRC,
        ensure_row_contiguous=True,
    )

    # Launch: grid over (DHQK/siz_b_DHQK, DHHV/siz_b_DHHV, B*NH)
    num_tiles_DHQK = (DHQK + siz_b_DHQK - 1) // siz_b_DHQK
    num_tiles_DHHV = (DHHV + siz_b_DHHV - 1) // siz_b_DHHV
    grid = (num_tiles_DHQK, num_tiles_DHHV, B * NH)
    threadgroup = (siz_b_DHHV, siz_b_DHQK, 1)

    outputs = kernel(
        inputs=[matQ, vecF, scaM_inter, vecM_combine, matDeltaH,
                vecN_out, matDeltaC_last, params, strides],
        output_shapes=[matDeltaC_states.shape],
        output_dtypes=[matQ.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )

    return outputs[0]


__all__ = ['mlstm_chunkwise_recurrent_bw_dC_metal']
