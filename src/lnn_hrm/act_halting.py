import torch
import torch.nn as nn


class ACTHaltingHead(nn.Module):
    """Adaptive Computation Time (ACT) halting head (single-step variant).

    For each token state h_t, predict halting logit and produce:
    - halt_prob: σ(logit) in [0,1]
    - halt_mask: (halt_prob > threshold)
    - stats: dict with mean probability and open rate
    This is a lightweight head to surface halting telemetry; it does not alter
    the sequence yet. Training code can add ponder loss externally.
    """

    def __init__(self, d_model: int, threshold: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.threshold = float(threshold)

    def forward(self, h: torch.Tensor):
        # h: (B, L, D)
        logits = self.proj(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        mask = probs > self.threshold
        stats = {
            "act_prob_mean": float(probs.mean().item()),
            "act_open_rate": float(mask.float().mean().item()),
        }
        return probs, mask, stats

