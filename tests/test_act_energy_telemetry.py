import torch


def test_act_and_energy_telemetry_ranges():
    from xlstm_official_full.xlstm_block_stack import xLSTMBlockStackConfig
    from xlstm_official_full.blocks.slstm.block import sLSTMBlockConfig
    from src.lnn_hrm.xlstm_hrm import HRMXLSTM

    B, L, D = 1, 8, 16
    slcfg = sLSTMBlockConfig(); slcfg.slstm.embedding_dim = D; slcfg.slstm.dropout = 0.0
    cfg = xLSTMBlockStackConfig(num_blocks=1, embedding_dim=D, dropout=0.0, slstm_block=slcfg, slstm_at="all")
    dev = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if dev.type == 'cpu':
        import pytest
        pytest.skip("sLSTM compiled backend requires GPU (MPS/CUDA)")
    model = HRMXLSTM(cfg).to(dev)
    x = torch.randn(B, L, D, device=dev)
    times = torch.arange(L).unsqueeze(0)
    with torch.no_grad():
        y, telem = model(x, times=times.to(dev))
    assert y.shape == x.shape
    assert 0.0 <= telem["alpha_mean"] <= 1.0
    assert 0.0 <= telem["conf_mean"] <= 1.0
    assert 0.0 <= telem["act_prob_mean"] <= 1.0
    assert 0.0 <= telem["act_open_rate"] <= 1.0
    # energy non-negative
    assert telem["energy_pre_gate"] >= 0.0 and telem["energy_post_gate"] >= 0.0
