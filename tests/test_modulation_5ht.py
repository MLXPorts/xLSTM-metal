import torch


def test_5ht_reduces_alpha_and_gain_cpu():
    from src.lnn_hrm.cube_gated_block import CubeGatedBlock
    B, L, D = 2, 10, 16
    h = torch.randn(B, L, D)
    y_teacher = h + 0.5
    times = torch.arange(L).unsqueeze(0).expand(B, -1)
    blk = CubeGatedBlock(d_in=D, d_key=D, d_val=D, fuse_phase_keys=False, k_5ht=0.8)
    blk.train()
    # No serotonin
    y0, a0, c0 = blk(h, y_teacher=y_teacher, train=True, allow_commit=None, times=times, mod_5ht=None)
    # High serotonin (divisive gain)
    mod = torch.ones(B, L)  # elevated 5-HT
    y1, a1, c1 = blk(h, y_teacher=y_teacher, train=True, allow_commit=None, times=times, mod_5ht=mod)
    assert a1 < a0 + 1e-6, (a0, a1)

