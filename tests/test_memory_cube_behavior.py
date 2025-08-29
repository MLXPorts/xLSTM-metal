import torch


def test_capacity_truncation_keeps_last_items():
    from src.lnn_hrm.memory_cube import MemoryCube

    cap = 5
    D = 3
    cube = MemoryCube(d_key=D, d_val=D, max_items=cap, topk=2)
    # Insert 8 distinct items with identifiable last column values 0..7
    k = torch.eye(D).repeat(8, 1)  # not used semantically; just shapes
    v = torch.zeros(8, D)
    v[:, -1] = torch.arange(8).float()
    with torch.no_grad():
        cube.update(k[:4], v[:4])
        cube.update(k[4:], v[4:])
    assert cube.keys.shape[0] == cap
    # Expect last 5 values: indices 3,4,5,6,7
    last_vals = cube.vals[:, -1].tolist()
    assert last_vals == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_confidence_increases_after_commit():
    from src.lnn_hrm.cube_gated_block import CubeGatedBlock

    B, L, D = 1, 5, 8
    h = torch.randn(B, L, D)
    times = torch.arange(L).unsqueeze(0)
    blk = CubeGatedBlock(d_in=D, d_key=D, d_val=D, fuse_phase_keys=False)

    # First pass: empty cube
    with torch.no_grad():
        y0, a0, c0 = blk(h, y_teacher=h, train=False, allow_commit=None, times=times)
    # Train pass: commit all positions
    allow_commit = torch.ones_like(times, dtype=torch.bool)
    blk.train()
    _y1, _a1, _c1 = blk(h, y_teacher=h + 1.0, train=True, allow_commit=allow_commit, times=times)
    # Eval pass: expect higher confidence with identical queries
    blk.eval()
    with torch.no_grad():
        y2, a2, c2 = blk(h, y_teacher=h, train=False, allow_commit=None, times=times)
    assert c2 > c0, f"confidence did not increase after commit: before={c0} after={c2}"

