import torch


def test_efficient_s4_layer_shapes():
    from ukm.temporal.s4_layer import EfficientS4Layer
    B, L, H = 2, 32, 64
    x = torch.randn(B, L, H)
    layer = EfficientS4Layer(d_model=H, seq_len=L, d_state=16, num_heads=8)
    with torch.no_grad():
        y = layer(x)
    assert tuple(y.shape) == (B, L, H)

