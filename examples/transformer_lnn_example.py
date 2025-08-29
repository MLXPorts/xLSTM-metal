"""
Minimal SolaceCore HRM+ hybrid demo on MPS.

Shows a tiny TransformerLNNHybrid running on Apple Silicon (MPS),
querying/updating a memory cube and reporting gate/confidence telemetry.
"""

import torch
from ukm.hrm.transformer_lnn import TransformerLNNHybrid


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    B, L, D = 2, 32, 64
    x = torch.randn(B, L, D, device=device)

    model = TransformerLNNHybrid(input_dim=D, hidden_dim=128, seq_len=L, cube_capacity=128).to(device)
    model.train()

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    print('Output:', tuple(out2['output'].shape), 'Device:', out2['output'].device)
    print('alpha_mean:', float(out2['alpha_mean']), 'conf:', float(out2['conf']))


if __name__ == '__main__':
    main()

