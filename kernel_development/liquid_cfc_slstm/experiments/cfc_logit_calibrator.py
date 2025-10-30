from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reversible_cell_experiment import CfCTorch


class CfCLogitCalibrator(nn.Module):
    """CfC-based per-step logit calibrator.

    Inputs per step: feature vector derived from logits (entropy, max prob, spread, token id).
    CfC core updates hidden state and emits a small control which we map to (temp_scale, bias).
    Apply as: logits' = logits / temp_scale + bias.
    Defaults initialize near identity (temp≈1, bias≈0).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 32,
        backbone_units: int = 64,
        backbone_layers: int = 1,
        mode: str = "default",
        activation: str = "lecun_tanh",
        topk_bias: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.topk_bias = topk_bias
        self.core = CfCTorch(input_size=4, units=hidden, mode=mode, backbone_units=backbone_units,
                             backbone_layers=backbone_layers, activation=activation)
        self.head = nn.Linear(hidden, 2)
        self.head_k = nn.Linear(hidden, topk_bias) if topk_bias and topk_bias > 0 else None
        # Initialize to identity: temp ≈ 1, bias ≈ 0
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias[:] = torch.tensor([0.0, 0.0])
        self.register_buffer("h0", torch.zeros(1, hidden))

    @staticmethod
    def _features_from_logits(logits: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        # logits: (B, V)
        probs = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * logp).sum(dim=-1, keepdim=True)  # (B,1)
        max_prob, _ = probs.max(dim=-1, keepdim=True)
        # spread: (p_max - p_2nd)
        top2 = torch.topk(probs, k=2).values
        spread = (top2[:, :1] - top2[:, 1:2])
        if token_ids is None:
            tok = torch.zeros_like(entropy)
        else:
            tok = (token_ids.float().unsqueeze(-1) / 50000.0).clamp(0, 1)
        return torch.cat([entropy, max_prob, spread, tok], dim=-1)

    def forward(self, logits: torch.Tensor, h: torch.Tensor | None, token_ids: torch.Tensor | None = None):
        """

        :param logits:
        :param h:
        :param token_ids:
        :return:
        """
        # logits: (B, V); h: (B, H) or None
        B = logits.size(0)
        feats = self._features_from_logits(logits, token_ids).unsqueeze(1)  # (B,1,4)
        h0 = self.h0.repeat(B, 1) if h is None else h
        hseq = self.core(feats, h0=h0)  # (B,1,H)
        h1 = hseq[:, -1, :]
        ctrl = self.head(h1)  # (B,2)
        temp_scale = F.softplus(ctrl[:, :1]) + 1e-3  # >0
        bias = ctrl[:, 1:2]
        logits_adj = logits / temp_scale + bias
        # Optional top-k sparse bias over current top tokens
        if self.topk_bias and self.topk_bias > 0 and self.head_k is not None:
            with torch.no_grad():
                topk = torch.topk(logits, k=min(self.topk_bias, logits.size(-1))).indices  # (B,K)
            k_bias = self.head_k(h1)  # (B,K)
            # Scatter-add into logits_adj
            logits_adj.scatter_add_(dim=-1, index=topk, src=k_bias)
        return logits_adj, h1
