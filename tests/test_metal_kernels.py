import pytest
import mlx.core as mx

from xlstm_metal.blocks.mlstm.kern import mlstm_chunkwise_recurrent_fw_C_metal
from xlstm_metal.blocks.mlstm.kern import mlstm_chunkwise_recurrent_bw_dC_metal
from xlstm_metal.blocks.mlstm.kern import mlstm_chunkwise_parallel_bw_dK_metal
from xlstm_metal.blocks.mlstm.kern import mlstm_chunkwise_parallel_bw_dQ_metal
from xlstm_metal.blocks.mlstm.kern import mlstm_chunkwise_parallel_bw_dV_metal
from xlstm_metal.blocks.mlstm.kern import mlstm_chunkwise_parallel_fw_Hintra_metal


@pytest.fixture
def setup_data():
    """

    :return:
    """
    B = 1;
    NH = 1;
    S = 32;
    DHQK = 16;
    DHHV = 16;
    NC = 2;
    L = 16
    qk_scale = 1.0;
    save_states_every_nth_chunk = 1
    rng_shapes = {
        'matQ': (B, NH, S, DHQK),
        'matK': (B, NH, S, DHQK),
        'matV': (B, NH, S, DHHV),
        'vecF': (B, NH, S),
        'vecI': (B, NH, S),
        'vecB': (B, NH, NC, L),
        'vecA': (B, NH, NC, L),
        'scaM_inter': (B, NH, NC + 1),
        'vecM_combine': (B, NH, S),
        'matDeltaH': (B, NH, S, DHHV),
        'vecN_out': (B, NH, S),
        'vecM_out': (B, NH, S),
        'matDeltaC_last': (B, NH, DHQK, DHHV),
        'matCstate_all': (B, NH, (NC + 1) * DHQK, DHHV),
        'vecNstate_all': (B, NH, (NC + 1) * DHQK),
        'scaMstate_all': (B, NH, NC + 1),
        'matDeltaH_out': (B, NH, S, DHHV),
        'matDeltaC_states': (B, NH, (NC + 1) * DHQK, DHHV),
    }
    tensors = {k: mx.random.normal(v) for k, v in rng_shapes.items()}
    tensors.update(dict(B=B, NH=NH, S=S, DHQK=DHQK, DHHV=DHHV, NC=NC, L=L, qk_scale=qk_scale,
                        save_states_every_nth_chunk=save_states_every_nth_chunk))
    return tensors


def test_fw_recurrent_kernel(setup_data):
    d = setup_data
    matC_states, vecN_states, scaMinter_states = mlstm_chunkwise_recurrent_fw_C_metal(
        d['matK'], d['matV'], d['vecF'], d['vecI'], None, None, None, d['NC'], d['L'])
    print("fw_recurrent sums", float(mx.sum(matC_states)), float(mx.sum(vecN_states)), float(mx.sum(scaMinter_states)))
    assert mx.sum(matC_states).item() != 0


def test_bw_recurrent_kernel(setup_data):
    d = setup_data
    out = mlstm_chunkwise_recurrent_bw_dC_metal(d['matQ'], d['vecF'], d['scaM_inter'], d['vecM_combine'],
                                                d['matDeltaH'], d['vecN_out'], d['matDeltaC_last'], d['NC'], d['L'],
                                                d['qk_scale'],
                                                save_states_every_nth_chunk=d['save_states_every_nth_chunk'])
    print("bw_recurrent sum", float(mx.sum(out)))
    assert mx.sum(out).item() != 0


def test_bw_parallel_dK_kernel(setup_data):
    d = setup_data
    matDeltaK = mlstm_chunkwise_parallel_bw_dK_metal(d['matQ'], d['matK'], d['matV'], d['vecI'], d['vecB'], d['vecA'],
                                                     d['matCstate_all'], d['vecNstate_all'], d['scaMstate_all'],
                                                     d['vecN_out'], d['vecM_out'], d['matDeltaH_out'],
                                                     d['matDeltaC_states'], d['NC'], d['L'], d['qk_scale'])
    print("bw_dK sum", float(mx.sum(matDeltaK)))
    assert mx.sum(matDeltaK).item() != 0


def test_bw_parallel_dQ_kernel(setup_data):
    d = setup_data
    matDeltaQ = mlstm_chunkwise_parallel_bw_dQ_metal(d['matQ'], d['matK'], d['matV'], d['vecI'], d['vecB'], d['vecA'],
                                                     d['matCstate_all'], d['vecNstate_all'], d['scaMstate_all'],
                                                     d['vecN_out'], d['vecM_out'], d['matDeltaH_out'],
                                                     d['matDeltaC_states'], d['NC'], d['L'], d['qk_scale'])
    print("bw_dQ sum", float(mx.sum(matDeltaQ)))
    assert mx.sum(matDeltaQ).item() != 0


def test_bw_parallel_dV_kernel(setup_data):
    d = setup_data
    matDeltaV = mlstm_chunkwise_parallel_bw_dV_metal(d['matQ'], d['matK'], d['matV'], d['vecI'], d['vecB'], d['vecA'],
                                                     d['matCstate_all'], d['vecNstate_all'], d['scaMstate_all'],
                                                     d['vecN_out'], d['vecM_out'], d['matDeltaH_out'],
                                                     d['matDeltaC_states'], d['NC'], d['L'], d['qk_scale'])
    print("bw_dV sum", float(mx.sum(matDeltaV)))
    assert mx.sum(matDeltaV).item() != 0


def test_fw_parallel_kernel(setup_data):
    d = setup_data
    matHout, vecNout, vecMout = mlstm_chunkwise_parallel_fw_Hintra_metal(d['matQ'], d['matK'], d['matV'],
                                                                         d['matCstate_all'], d['vecNstate_all'],
                                                                         d['scaMstate_all'], d['vecI'], d['vecB'],
                                                                         d['NC'], d['L'], d['qk_scale'])
    print("fw_parallel sums", float(mx.sum(matHout)), float(mx.sum(vecNout)), float(mx.sum(vecMout)))
    assert mx.sum(matHout).item() != 0
