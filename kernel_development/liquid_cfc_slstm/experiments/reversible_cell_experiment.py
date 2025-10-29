from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def device_auto() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_vocab():
    vowels = set("AEIOU")
    punctuation = set("!?.,;")
    consonants = set("BCDFGHJKLMNPQRSTVWXYZ")
    return vowels, consonants, punctuation


def generate_batch(batch: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (input_codes, labels) where labels are shape (B, S, 2): vowel?, punct?"""
    vowels, consonants, punctuation = make_vocab()
    import random

    xs = []
    ys = []
    for _ in range(batch):
        word = []
        labels = []
        for i in range(seq_len - 1):
            if random.random() < 0.3:
                ch = random.choice(tuple(vowels))
            else:
                ch = random.choice(tuple(consonants))
            code = ord(ch)
            word.append(code)
            labels.append([1.0 if ch in vowels else 0.0, 0.0])
        # punctuation last
        p = random.choice(tuple(punctuation))
        word.append(ord(p))
        labels.append([0.0, 1.0])
        xs.append(word)
        ys.append(labels)
    x = torch.tensor(xs, dtype=torch.float32, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device)
    return x, y


class SLSTMHead(nn.Module):
    """Minimal stabilized sLSTM-like head for toy task.

    Exponential gates with a running normalizer n; readout uses sigmoid(c).
    """

    def __init__(self, hidden: int, input_scale: float = 100.0):
        super().__init__()
        self.hidden = hidden
        self.W_i = nn.Linear(1, hidden)
        self.W_f = nn.Linear(1, hidden)
        self.W_o = nn.Linear(1, hidden)
        self.W_g = nn.Linear(1, hidden)
        self.proj = nn.Linear(hidden, 2)
        self.register_buffer("n_init", torch.zeros(1, hidden))
        self.input_scale = input_scale

    def forward(self, x_codes: torch.Tensor) -> torch.Tensor:
        # x: (B, S)
        B, S = x_codes.shape
        h = torch.zeros(B, self.hidden, device=x_codes.device)
        c = torch.zeros_like(h)
        n = self.n_init.repeat(B, 1)
        outs = []
        x = (x_codes - 65.0) / self.input_scale
        for t in range(S):
            xt = x[:, t:t+1]
            pre_i = self.W_i(xt)
            pre_f = self.W_f(xt)
            pre_o = self.W_o(xt)
            pre_g = self.W_g(xt)
            i_t = torch.exp(pre_i - n)
            f_t = torch.exp(pre_f - n)
            o_t = torch.exp(pre_o - n)
            g_t = torch.sigmoid(pre_g)
            c = f_t * c + i_t * g_t
            n = n + 1e-2 * (i_t + f_t + o_t - 3.0)
            h = torch.sigmoid(c) * o_t
            outs.append(self.proj(h))
        return torch.stack(outs, dim=1)


class CfCHead(nn.Module):
    """Continuous-time smoothing head: exp gates + CfC h update.

    act_c: activation applied to c_new in the feed-forward term ("sigmoid" or "lecun_tanh")
    act_g: activation for candidate g preact ("sigmoid" or "lecun_tanh")
    """

    def __init__(self, hidden: int, input_scale: float = 100.0, act_c: str = "sigmoid", act_g: str = "sigmoid"):
        super().__init__()
        self.hidden = hidden
        self.W_i = nn.Linear(1, hidden)
        self.W_f = nn.Linear(1, hidden)
        self.W_o = nn.Linear(1, hidden)
        self.W_g = nn.Linear(1, hidden)
        self.W_lambda = nn.Linear(1, hidden)
        self.proj = nn.Linear(hidden, 2)
        self.register_buffer("n_init", torch.zeros(1, hidden))
        self.input_scale = input_scale
        self.act_c = act_c
        self.act_g = act_g

    @staticmethod
    def lecun_tanh(x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh(0.666 * x)

    def forward(self, x_codes: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        B, S = x_codes.shape
        h = torch.zeros(B, self.hidden, device=x_codes.device)
        c = torch.zeros_like(h)
        n = self.n_init.repeat(B, 1)
        outs = []
        x = (x_codes - 65.0) / self.input_scale
        for t in range(S):
            xt = x[:, t:t+1]
            pre_i = self.W_i(xt)
            pre_f = self.W_f(xt)
            pre_o = self.W_o(xt)
            pre_g = self.W_g(xt)
            lam = torch.sigmoid(self.W_lambda(xt)) * 1.0  # [0,1]
            i_t = torch.exp(pre_i - n)
            f_t = torch.exp(pre_f - n)
            o_t = torch.exp(pre_o - n)
            if self.act_g == "lecun_tanh":
                g_t = self.lecun_tanh(pre_g)
            else:
                g_t = torch.sigmoid(pre_g)
            c = f_t * c + i_t * g_t
            if self.act_c == "lecun_tanh":
                ff = o_t * self.lecun_tanh(c)
            else:
                ff = o_t * torch.sigmoid(c)
            h = (h + dt * ff) / (1.0 + dt * lam)
            n = n + 1e-2 * (i_t + f_t + o_t - 3.0)
            outs.append(self.proj(h))
        return torch.stack(outs, dim=1)


class ReversibleTauAccumulator(nn.Module):
    """Reversible-flavored accumulator with input-dependent tau and softsign update."""

    def __init__(self, hidden: int, env: int = 5, input_scale: float = 100.0):
        super().__init__()
        self.hidden = hidden
        self.env = env
        self.W_in = nn.Linear(1, hidden)
        self.W_h = nn.Linear(hidden, hidden, bias=True)
        nn.init.eye_(self.W_h.weight)
        self.W_env = nn.Linear(hidden, env)
        self.W_out = nn.Linear(env, 2)
        self.tau_base = nn.Parameter(torch.tensor(0.01))
        self.tau_w = nn.Parameter(torch.tensor(0.4))
        self.tau_scale = nn.Parameter(torch.tensor(100.0))
        self.input_scale = input_scale

    @staticmethod
    def softsign(x: torch.Tensor) -> torch.Tensor:
        return x / (1 + x.abs())

    def compute_tau(self, xt: torch.Tensor) -> torch.Tensor:
        # xt: (B,1) scaled input
        return torch.sigmoid(self.tau_base + torch.tanh(self.tau_w * xt))

    def forward(self, x_codes: torch.Tensor) -> torch.Tensor:
        B, S = x_codes.shape
        h = torch.zeros(B, self.hidden, device=x_codes.device)
        outs = []
        x = (x_codes - 65.0) / self.input_scale
        for t in range(S):
            xt = x[:, t:t+1]
            tau = self.compute_tau((xt - 0.0) / self.tau_scale)
            h_update = self.W_in(xt) + self.W_h(h)
            h = (1 - tau) * h + tau * self.softsign(h_update)
            e = F.relu(self.W_env(h))
            outs.append(self.W_out(e))
        return torch.stack(outs, dim=1)


class CfCTorch(nn.Module):
    """PyTorch translation of CfC cell (default/no_gate/pure) with optional backbone.

    This mirrors the public CfC cell structure (Lechner & Hasani), using a small
    MLP backbone over [x, h] to produce ff1/ff2 and time_a/time_b. Mode:
      - default: new_h = ff1*(1 - t_interp) + t_interp*ff2
      - no_gate: new_h = ff1 + t_interp*ff2
      - pure:    new_h = -A * exp(-t*(|w_tau| + |ff1|)) * ff1 + A
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        mode: str = "default",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        activation: str = "lecun_tanh",
        input_scale: float = 100.0,
    ):
        super().__init__()
        assert mode in ("default", "no_gate", "pure")
        self.units = units
        self.mode = mode
        self.input_size = input_size
        self.input_scale = input_scale
        self.back_layers = nn.ModuleList()
        feat_dim = input_size + units
        if backbone_layers > 0 and backbone_units > 0:
            dim = feat_dim
            for i in range(backbone_layers):
                self.back_layers.append(nn.Linear(dim, backbone_units))
                self.back_layers.append(nn.Dropout(backbone_dropout))
                dim = backbone_units
            feat_dim = backbone_units
        # Output heads
        self.ff1 = nn.Linear(feat_dim, units)
        if mode != "pure":
            self.ff2 = nn.Linear(feat_dim, units)
            self.time_a = nn.Linear(feat_dim, units)
            self.time_b = nn.Linear(feat_dim, units)
        else:
            self.w_tau = nn.Parameter(torch.zeros(1, units))
            self.A = nn.Parameter(torch.ones(1, units))
        self._act = activation

    @staticmethod
    def lecun_tanh(x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh(0.666 * x)

    def _backbone(self, feat: torch.Tensor) -> torch.Tensor:
        x = feat
        for layer in self.back_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self._act == "lecun_tanh":
                    x = self.lecun_tanh(x)
                else:
                    x = torch.tanh(x)
            else:
                x = layer(x)
        return x

    def forward(self, x_codes: torch.Tensor, h0: torch.Tensor | None = None, timespans: torch.Tensor | None = None) -> torch.Tensor:
        # x_codes: (B,S,input_size=1) or (B,S)
        if x_codes.dim() == 2:
            x_codes = x_codes.unsqueeze(-1)
        B, S, C = x_codes.shape
        device = x_codes.device
        h = torch.zeros(B, self.units, device=device) if h0 is None else h0
        outs = []
        xs = (x_codes - 65.0) / self.input_scale
        for t in range(S):
            xt = xs[:, t, :]
            feat = torch.cat([xt, h], dim=-1)
            f = self._backbone(feat) if len(self.back_layers) > 0 else feat
            ff1 = self.ff1(f)
            if self.mode == "pure":
                tau_term = torch.abs(self.w_tau) + torch.abs(ff1)
                tspan = 1.0 if timespans is None else timespans[:, t].reshape(-1, 1)
                new_h = -self.A * torch.exp(-tspan * tau_term) * ff1 + self.A
            else:
                ff2 = self.ff2(f)
                t_a = self.time_a(f)
                t_b = self.time_b(f)
                tspan = 1.0 if timespans is None else timespans[:, t].reshape(-1, 1)
                t_interp = torch.sigmoid(-t_a * tspan + t_b)
                if self.mode == "no_gate":
                    new_h = ff1 + t_interp * ff2
                else:
                    new_h = ff1 * (1.0 - t_interp) + t_interp * ff2
            h = new_h
            outs.append(h)
        return torch.stack(outs, dim=1)


@dataclass
class TrainConfig:
    seq_len: int = 9
    hidden: int = 32
    env: int = 5
    epochs: int = 2
    batches_per_epoch: int = 40
    batch_size: int = 32
    lr: float = 1e-3


def train_one(model: nn.Module, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_hist = []
    for _ in range(cfg.epochs):
        for _ in range(cfg.batches_per_epoch):
            x, y = generate_batch(cfg.batch_size, cfg.seq_len, device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_hist.append(loss.item())
    return {"loss_mean": float(sum(loss_hist) / max(1, len(loss_hist))), "loss_last": float(loss_hist[-1])}


def quick_eval(model: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x, y = generate_batch(64, 9, device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds.eq(y).float().mean()).item()
    return {"acc": acc}
