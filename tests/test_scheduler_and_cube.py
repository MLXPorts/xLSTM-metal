import torch


def test_boundary_commit_mask_and_cube_update_respect_mask():
    from src.lnn_hrm.scheduler import boundary_commit_mask
    from src.lnn_hrm.cube_gated_block import CubeGatedBlock

    B, L, D = 1, 10, 8
    h = torch.randn(B, L, D)
    times = torch.arange(L).unsqueeze(0)
    commit = boundary_commit_mask(times)  # True where slot==4 -> positions 4,9,...
    block = CubeGatedBlock(d_in=D, d_key=D, d_val=D, fuse_phase_keys=False)

    # Before any updates, cube is empty
    assert block.cube.keys.numel() == 0

    # Teacher different from h to produce nonzero deltas
    y_teacher = h + 1.0
    block.train()
    _y, a, c = block(h, y_teacher=y_teacher, train=True, allow_commit=commit, times=times)
    # Expect updates only at commit positions: count True in commit
    expected = int(commit.sum().item()) * B
    assert block.cube.keys.shape[0] == expected


def test_phase_key_fusion_shapes_and_forward():
    from src.lnn_hrm.cube_gated_block import CubeGatedBlock
    B, L, D = 2, 7, 12
    h = torch.randn(B, L, D)
    times = torch.arange(L).unsqueeze(0).expand(B, -1)
    blk = CubeGatedBlock(d_in=D, d_key=D, d_val=D, fuse_phase_keys=True)
    y, a, c = blk(h, y_teacher=h, train=False, allow_commit=None, times=times)
    assert y.shape == h.shape
    assert 0.0 <= a <= 1.0
    assert 0.0 <= c <= 1.0

