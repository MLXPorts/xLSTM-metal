import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return (x / rms) * self.weight


class StabilizedXLSTMCell(nn.Module):
    """
    Stabilized xLSTM cell with soft-capped gates and running normalizer.
    Matches the math in the comparison_proofs doc.
    """

    def __init__(self, d_model: int, gate_softcap: float = 15.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.gate_softcap = gate_softcap
        self.W_i = nn.Linear(d_model, d_model, bias=True)
        self.W_f = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.W_z = nn.Linear(d_model, d_model, bias=True)
        # Bias initialization per spec
        with torch.no_grad():
            self.W_i.bias.fill_(-10.0)
            self.W_f.bias.fill_(+1.0)
        self.rms = RMSNorm(d_model)

    @staticmethod
    def softcap(x: torch.Tensor, a: float) -> torch.Tensor:
        return a * torch.tanh(x / a)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor, n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = self.rms(x_t)
        i_tilde = self.W_i(x_t)
        f_tilde = self.W_f(x_t)
        o_tilde = self.W_o(x_t)
        z_tilde = self.W_z(x_t)
        i_cap = self.softcap(i_tilde, self.gate_softcap)
        f_cap = self.softcap(f_tilde, self.gate_softcap)
        i = torch.exp(i_cap)
        f = torch.sigmoid(f_cap)
        o = torch.sigmoid(o_tilde)
        z = torch.tanh(z_tilde)
        c = f * c + i * z
        n = f * n + i
        h = o * (c / (n + 1e-8))
        return h, c, n


class StabilizedXLSTM(nn.Module):
    """Stacked xLSTM with projection in/out."""

    def __init__(self, d_model: int, num_layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList([StabilizedXLSTMCell(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        n = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            outs = []
            h_l, c_l, n_l = h, c, n
            for t in range(L):
                h_l, c_l, n_l = layer(x[:, t, :], h_l, c_l, n_l)
                outs.append(h_l)
            x = torch.stack(outs, dim=1)
            h, c, n = h_l, c_l, n_l
        return x


class RobustScaledSTE(nn.Module):
    """Robust Scaled Straight-Through Estimator for weight quantization."""

    def __init__(self, threshold: float = 0.7, gradient_scale: float = 0.1) -> None:
        super().__init__()
        self.threshold = threshold
        self.gradient_scale = gradient_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, -10.0, 10.0)
        with torch.no_grad():
            q = torch.where(x_safe > self.threshold, torch.ones_like(x_safe),
                            torch.where(x_safe < -self.threshold, -torch.ones_like(x_safe), torch.zeros_like(x_safe)))
        # Scaled STE gradient
        return x_safe * (1 - self.gradient_scale) + (q - x_safe * (1 - self.gradient_scale)).detach()


class ConservativeSTE(nn.Module):
    def __init__(self, threshold: float = 0.9) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, -6.0, 6.0)
        with torch.no_grad():
            q = torch.where(x_safe > self.threshold, torch.ones_like(x_safe),
                            torch.where(x_safe < -self.threshold, -torch.ones_like(x_safe), torch.zeros_like(x_safe)))
        return x_safe * 0.5 + (q - x_safe * 0.5).detach()


class QuantizedXLSTMCell(StabilizedXLSTMCell):
    def __init__(self, d_model: int, ste: nn.Module, quantize_weights: bool = True) -> None:
        super().__init__(d_model)
        self.quantize_weights = quantize_weights
        self.ste = ste

    def forward(self, x_t, h, c, n):
        x_t = self.rms(x_t)
        if self.quantize_weights:
            Wi = self.ste(self.W_i.weight); Wf = self.ste(self.W_f.weight)
            Wo = self.ste(self.W_o.weight); Wz = self.ste(self.W_z.weight)
            i_tilde = F.linear(x_t, Wi, self.W_i.bias)
            f_tilde = F.linear(x_t, Wf, self.W_f.bias)
            o_tilde = F.linear(x_t, Wo, self.W_o.bias)
            z_tilde = F.linear(x_t, Wz, self.W_z.bias)
        else:
            i_tilde = self.W_i(x_t); f_tilde = self.W_f(x_t)
            o_tilde = self.W_o(x_t); z_tilde = self.W_z(x_t)
        i_cap = self.softcap(i_tilde, self.gate_softcap)
        f_cap = self.softcap(f_tilde, self.gate_softcap)
        i = torch.exp(i_cap); f = torch.sigmoid(f_cap)
        o = torch.sigmoid(o_tilde); z = torch.tanh(z_tilde)
        c = f * c + i * z
        n = f * n + i
        h = o * (c / (n + 1e-8))
        return h, c, n

