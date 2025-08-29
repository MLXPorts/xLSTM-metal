import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LiquidTimeConstantCell(nn.Module):
    """
    Stable liquid-like recurrent cell (simple LTC/CfC-inspired).

    Continuous-time update (discretized):
        h_t = h_{t-1} + dt * ( -h_{t-1} / tau + phi(Wx x_t + Wh h_{t-1} + b) )
    with learnable tau>0 and dt in (0, dt_max) via sigmoid gate.

    Optional blocky (quantized) activation with straight-through estimator.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt_max: float = 0.2,
        tau_init: float = 1.5,
        blocky_levels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt_max = dt_max
        self.blocky_levels = blocky_levels

        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Positive tau via softplus
        self.tau_raw = nn.Parameter(torch.full((hidden_dim,), tau_init))
        # Step-size gate
        self.dt_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _phi(self, u: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(u)
        if self.blocky_levels is None or self.blocky_levels <= 1:
            return z
        # Quantize to L levels in [-1,1] with STE
        L = float(self.blocky_levels - 1)
        with torch.no_grad():
            q = torch.round((z + 1) * (L / 2.0)) / (L / 2.0) - 1.0
            q = torch.clamp(q, -1.0, 1.0)
        return z + (q - z).detach()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        u = self.Wx(x_t) + self.Wh(h_prev) + self.bias
        phi = self._phi(u)
        tau = F.softplus(self.tau_raw) + 1e-3  # ensure strictly positive
        gate_inp = torch.cat([x_t, h_prev], dim=-1)
        dt = torch.sigmoid(self.dt_gate(gate_inp)) * self.dt_max
        dh = -h_prev / tau + phi
        h = h_prev + dt * dh
        return h


class LiquidBlock(nn.Module):
    """Sequence wrapper for LiquidTimeConstantCell."""

    def __init__(self, input_dim: int, hidden_dim: int, **cell_kwargs) -> None:
        super().__init__()
        self.cell = LiquidTimeConstantCell(input_dim, hidden_dim, **cell_kwargs)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, device=x.device, dtype=x.dtype) if h0 is None else h0
        outs = []
        for t in range(L):
            h = self.cell(x[:, t, :], h)
            outs.append(h)
        H = torch.stack(outs, dim=1)
        return self.norm(H)

