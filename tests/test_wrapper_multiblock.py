import torch


def test_wrapper_multiblock_updates_conf_on_second_pass():
    from xlstm_official_full.xlstm_block_stack import xLSTMBlockStackConfig
    from xlstm_official_full.blocks.slstm.block import sLSTMBlockConfig
    from src.lnn_hrm.xlstm_hrm import HRMXLSTM

    B, L, D = 1, 12, 16
    sl = sLSTMBlockConfig(); sl.slstm.embedding_dim = D; sl.slstm.dropout = 0.0
    cfg = xLSTMBlockStackConfig(num_blocks=3, embedding_dim=D, dropout=0.0, slstm_block=sl, slstm_at="all")

    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS required in this repo configuration")
    dev = torch.device('mps')
    model = HRMXLSTM(cfg).to(dev)
    x = torch.randn(B, L, D, device=dev)
    times = torch.arange(L, device=dev).unsqueeze(0)

    # First (train) pass to allow commits at boundaries
    model.train()
    y1, t1 = model(x, times=times)
    # Second (eval) pass: should read from cube; conf should not decrease
    model.eval()
    with torch.no_grad():
        y2, t2 = model(x, times=times)
    assert y1.shape == y2.shape == x.shape
    for k in ["alpha_mean", "act_prob_mean", "act_open_rate"]:
        assert 0.0 <= float(t2[k]) <= 1.0
    assert t2["energy_pre_gate"] >= 0.0 and t2["energy_post_gate"] >= 0.0
    assert float(t2["conf_mean"]) >= float(t1["conf_mean"]) - 1e-6
