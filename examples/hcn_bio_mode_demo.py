"""
HCN bio-mode vs neutral-mode demo

Demonstrates how a simple bio-inspired modulation (using HCN resonance and V½)
could affect a comb-like gate: when bio-mode is on, we boost an auxiliary
confidence according to a theta resonance kernel centered at f0=6 Hz.

Outputs:
- PNG plots under outputs/hcn_bio/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

os.makedirs("outputs/hcn_bio", exist_ok=True)


def hcn_resonance(f_hz: float, f0: float = 6.0, sigma: float = 4.0) -> float:
    return float(np.exp(-((f_hz - f0) ** 2) / (2.0 * sigma * sigma)))


class BioGate(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x, conf_aux=0.0):
        a = torch.sigmoid(self.mlp(x))
        # fold in auxiliary confidence as a small additive boost (clipped)
        a = (a + conf_aux).clamp(0, 1)
        return a


def run_demo(T_steps: int = 600, hidden_dim: int = 64, seed: int = 11):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    H = torch.from_numpy(rng.normal(0, 1, size=(T_steps, hidden_dim)).astype(np.float32))
    dY = torch.from_numpy(rng.normal(0, 0.2, size=(T_steps, hidden_dim)).astype(np.float32))
    base_conf = torch.from_numpy(rng.uniform(0.3, 0.8, size=(T_steps, 1)).astype(np.float32))

    gate = BioGate(d_in=hidden_dim * 2 + 1)

    # Frequency sweep over time to mimic changing theta power
    freqs = np.linspace(2.0, 12.0, T_steps)  # sweep through delta→alpha bands

    alpha_bio, alpha_neutral = [], []
    conf_bio_aux = []

    for t in range(T_steps):
        x = torch.cat([H[t], dY[t], base_conf[t]], dim=0)
        # bio-mode computes an auxiliary confidence from resonance
        r = hcn_resonance(freqs[t])
        conf_aux = 0.15 * r  # modest boost
        a_bio = gate(x, conf_aux=conf_aux)
        a_neu = gate(x, conf_aux=0.0)
        alpha_bio.append(a_bio.item())
        alpha_neutral.append(a_neu.item())
        conf_bio_aux.append(conf_aux)

    t = np.arange(T_steps)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, freqs, label='freq (Hz)')
    axs[0].set_ylabel('Hz')
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(t, conf_bio_aux, label='bio auxiliary conf', color='purple')
    axs[1].set_ylabel('aux conf')
    axs[1].grid(True, alpha=0.3)
    axs[2].plot(t, alpha_neutral, label='alpha neutral')
    axs[2].plot(t, alpha_bio, label='alpha bio-mode')
    axs[2].set_ylabel('alpha')
    axs[2].set_xlabel('time')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join('outputs/hcn_bio', 'hcn_bio_vs_neutral.png')
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    run_demo()

