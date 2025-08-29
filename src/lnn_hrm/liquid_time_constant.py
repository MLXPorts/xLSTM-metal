import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidTimeConstant(nn.Module):
    """Stable liquid cell (LTC-style) for block-scale updates.

    Batch-first API: x [B,D], h [B,D], t scalar or [B]. Returns (new_state, output).
    """

    def __init__(self, input_size: int, hidden_size: int, tau_init: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_net = nn.Linear(hidden_size, hidden_size)
        self.state_net_g = nn.Linear(hidden_size, hidden_size)
        self.state_net_h = nn.Linear(hidden_size, hidden_size)
        # strictly positive τ via softplus
        self.tau_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(tau_init)) - 1.0) * torch.ones(hidden_size))
        self.A = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor):
        if t.dim() == 0:
            t = t.expand(x.size(0))
        tau = F.softplus(self.tau_raw) + 1e-4
        comb = torch.cat([x, h], dim=-1)
        feats = self.backbone(comb)
        f_t = torch.sigmoid(self.time_net(feats))
        g_x = self.state_net_g(feats)
        h_x = self.state_net_h(feats)
        gate = torch.sigmoid(-f_t * t.view(-1, 1))
        h_new = gate * g_x + (1.0 - gate) * h_x
        # residual clamp for stability
        delta = (h_new - h).clamp(min=-1.0, max=1.0)
        h_out = self.norm(h + delta)
        return h_out, h_out

