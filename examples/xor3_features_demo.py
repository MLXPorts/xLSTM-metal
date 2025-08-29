"""
XOR3/Phase/Leak-Reminder Gate Demo

Shows how phase buckets (phi), a lightweight XOR3 balanced-ternary projection, and a
remainder feature r(t)=1-exp(-t/tau_ms) modulate a simple gate over synthetic streams.

Outputs:
- PNG plots under outputs/xor3_demo/
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

os.makedirs("outputs/xor3_demo", exist_ok=True)


def phase_bucket(time_ms: float, cycle_ms: float = 167.0) -> int:
    t = time_ms % cycle_ms
    bucket = int((t / cycle_ms) * 3.0)
    return max(0, min(2, bucket))


def phi_onehot(phi: int) -> torch.Tensor:
    v = torch.zeros(3)
    v[phi] = 1.0
    return v


def xor3_feature(h: torch.Tensor, thresh: float = 0.0) -> torch.Tensor:
    """Project hidden vector h to a 3-dim balanced-ternary-like feature.
    Here we simply take three chunked means and map via sign to {-1,0,+1}.
    """
    D = h.numel()
    thirds = max(1, D // 3)
    chunks = [h[i * thirds : (i + 1) * thirds] for i in range(3)]
    # pad if needed
    while len(chunks) < 3:
        chunks.append(torch.zeros(thirds))
    means = torch.stack([c.mean() if c.numel() > 0 else torch.tensor(0.0) for c in chunks])
    feat = torch.zeros(3)
    for i, m in enumerate(means):
        if m > thresh:
            feat[i] = 1.0
        elif m < -thresh:
            feat[i] = -1.0
        else:
            feat[i] = 0.0
    return feat


def remainder_feature(t_ms: float, tau_ms: float = 20.0) -> float:
    return 1.0 - math.exp(-t_ms / tau_ms)


class TinyGate(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        a = torch.sigmoid(self.net(x))
        return a.clamp(0, 1)


def run_demo(T_steps: int = 300, hidden_dim: int = 48, seed: int = 1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # synthetic hidden stream h_t and a teacher residual dY_t
    H = torch.from_numpy(rng.normal(0, 1, size=(T_steps, hidden_dim)).astype(np.float32))
    dY = torch.from_numpy(rng.normal(0, 0.25, size=(T_steps, hidden_dim)).astype(np.float32))
    conf = torch.from_numpy(rng.uniform(0.2, 0.9, size=(T_steps, 1)).astype(np.float32))
    novelty = 1.0 - conf

    gate = TinyGate(d_in=hidden_dim + hidden_dim + 3 + 3 + 1)

    alphas, phis, remainders = [], [], []
    blended_norms, teacher_norms = [], []

    for t in range(T_steps):
        h_t = H[t]
        dY_t = dY[t]
        t_ms = float(t)  # treat step index as ms for demo
        phi = phase_bucket(t_ms)
        phi_oh = phi_onehot(phi)
        xor_feat = xor3_feature(h_t)
        rem = remainder_feature(t_ms)
        inp = torch.cat([h_t, dY_t, conf[t], novelty[t], phi_oh, xor_feat, torch.tensor([rem])], dim=0)
        alpha = gate(inp)
        y_teacher = h_t + dY_t
        y_blend = (1 - alpha) * y_teacher + alpha * (h_t + 0.5 * dY_t)  # simple perturbation

        alphas.append(alpha.item())
        phis.append(phi)
        remainders.append(rem)
        blended_norms.append(y_blend.norm().item())
        teacher_norms.append(y_teacher.norm().item())

    # plots
    t = np.arange(T_steps)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, alphas, label='alpha')
    axs[0].set_ylabel('alpha')
    axs[0].grid(True, alpha=0.3)
    axs[1].step(t, phis, where='post', label='phi')
    axs[1].plot(t, remainders, label='remainder(t)', alpha=0.8)
    axs[1].set_ylabel('phi / remainder')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    axs[2].plot(t, teacher_norms, label='||y_teacher||')
    axs[2].plot(t, blended_norms, label='||y_blend||')
    axs[2].set_ylabel('norms')
    axs[2].set_xlabel('time step (ms)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join('outputs/xor3_demo', 'xor3_phi_remainder_gate.png')
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    run_demo()

