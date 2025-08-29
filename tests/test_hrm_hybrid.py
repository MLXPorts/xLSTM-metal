import torch


def test_transformer_lnn_hybrid_forward_cpu():
    from ukm.hrm.transformer_lnn import TransformerLNNHybrid
    B, L, D = 2, 16, 32
    x = torch.randn(B, L, D)
    model = TransformerLNNHybrid(input_dim=D, hidden_dim=64, seq_len=L, cube_capacity=64)
    model.train()
    out = model(x)
    assert 'output' in out and 'alpha_mean' in out and 'conf' in out
    y = out['output']
    assert tuple(y.shape) == (B, L, D)
    assert 0.0 <= float(out['alpha_mean']) <= 1.0
    assert 0.0 <= float(out['conf']) <= 1.0

