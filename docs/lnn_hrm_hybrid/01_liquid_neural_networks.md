# Liquid Neural Networks (LTC/CfC): Dynamics, Stability, and Practical Variants

This document summarizes liquid cell dynamics (LTC/CfC-style), closed-form updates used in practice, and stability conditions. We include a working PyTorch cell suitable for the hybrid.

---

## 1. Continuous-time dynamics

We consider per-neuron dynamics with positive time constants and input-conditioned drift:

dx/dt = -[(1/τ) + f(x, I, t; θ)] ⊙ x + f(x, I, t; θ) ⊙ A

where τ > 0 elementwise; f is a neural function (bounded/Lipschitz in practice); A is a bias-like vector.

For small Δt we can write a fused discrete step:

x_{t+Δt} = [x_t + Δt · f(x_t, I_t; θ) ⊙ A] / [1 + Δt · ((1/τ) + f(x_t, I_t; θ))]

We also consider a gating form that blends short-/long-term transforms (g,h):

x(t) = σ(−f t) ⊙ g(x,I;θ_g) + (1 − σ(−f t)) ⊙ h(x,I;θ_h)

This form is useful for stable, explicit updates without ODE solvers.

---

## 2. Boundedness and stability

- τ_i > 0 ensures intrinsic decay; effective τ_sys is reduced but remains positive:
  
  τ_sys_i = τ_i / (1 + τ_i f_i) ∈ (0, τ_i]

- If f is bounded/Lipschitz and weights are norm-bounded, one can construct Lyapunov functions V(x)=||x||^2 showing dV/dt ≤ 0 under mild conditions.

- In discrete practice, we further enforce:
  - Softplus on τ̂ to get τ = softplus(τ̂) + ε
  - Spectral or weight clipping
  - Residual norm clamp per step
  - fp32 state, bf16 activations

---

## 3. Practical PyTorch cell (reference)

Below is a compact, stable liquid cell for block-scale usage (batch-first). It mirrors your working concept and adds safety.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class LiquidTimeConstant(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, tau_init: float = 1.0):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.time_net = nn.Linear(hidden_size, hidden_size, bias=True)
        self.state_net_g = nn.Linear(hidden_size, hidden_size, bias=True)
        self.state_net_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.tau_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(tau_init)) - 1.0) * torch.ones(hidden_size))
        self.A = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor):
        # x: [B,D], h: [B,D], t: [B] or scalar
        if t.dim() == 0:
            t = t.expand(x.size(0))
        tau = F.softplus(self.tau_raw) + 1e-4
        comb = torch.cat([x, h], dim=-1)
        feats = self.backbone(comb)
        f_t = torch.sigmoid(self.time_net(feats))   # in (0,1)
        g_x = self.state_net_g(feats)
        h_x = self.state_net_h(feats)
        gate = torch.sigmoid(-f_t * t.view(-1, 1))
        h_new = gate * g_x + (1.0 - gate) * h_x
        # Stabilize: residual clamp
        delta = (h_new - h).clamp(min=-1.0, max=1.0)
        h_out = self.norm(h + delta)
        return h_out, h_out
```

Notes:
- The layer norm and residual clamp limit drift. In production we may use a more faithful closed-form fraction update.
- Replace `t` with learned or scheduled times per inner step.

---

## 4. CfC-style variant (optional)

Closed-form Continuous-time (CfC) cells compute analytical state interpolation between g,h under a learned f and time horizon; our gate form captures the core behavior while remaining compile-friendly.

---

## 5. Scheduling for HRM

- Within each HRM cycle, apply T liquid steps with fixed H context; then update H once.
- Reset L’s short-term memory between cycles (zero or learned reset) to create distinct phases.

---

## 6. Numerical guidelines

- Dtypes: params fp16/bf16; activations bf16; states fp32.
- Clamp step residuals |Δ| ≤ 1.0; optionally scale by adaptive norm.
- Use LeCun normal inits; RMSNorm/post-norm patterns for stability.

---

## 7. Testing checklist

- Determinism under canonical mode (fixed seeds, chunk size).
- Monotone decrease in validation loss for a simple synthetic task.
- PR (participation ratio) growth for H vs L on multi-task training.

