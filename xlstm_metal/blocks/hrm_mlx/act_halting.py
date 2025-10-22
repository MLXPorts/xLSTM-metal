#!/usr/bin/env python
"""
Adaptive Computation Time (ACT) Halting Head for MLX.

Predicts per-token halting probabilities for adaptive computation budgeting.
Ported from src/lnn_hrm/act_halting.py (PyTorch) to MLX.
"""

import mlx.core as mx
import mlx.nn as nn


class ACTHaltingHeadMLX(nn.Module):
    """Adaptive Computation Time (ACT) halting head.

    For each token state h_t, predicts halting logit and produces:
    - halt_prob: σ(logit) in [0,1]
    - halt_mask: (halt_prob > threshold)
    - stats: dict with mean probability and open rate

    This is a lightweight head to surface halting telemetry; it does not alter
    the sequence yet. Training code can add ponder loss externally.

    Args:
        d_model: Model dimension (hidden size)
        threshold: Default threshold for halting (default: 0.5)

    Example:
        >>> head = ACTHaltingHeadMLX(d_model=512, threshold=0.5)
        >>> h = mx.random.normal((2, 10, 512))  # (B, L, D)
        >>> probs, mask, stats = head(h)
        >>> print(f"Mean halt prob: {stats['act_prob_mean']:.3f}")
        >>> print(f"Halted tokens: {stats['act_open_rate']:.3f}")
    """

    def __init__(self, d_model: int, threshold: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.threshold = threshold

    def __call__(
        self,
        h: mx.array,
        threshold: float | None = None
    ) -> tuple[mx.array, mx.array, dict[str, float]]:
        """Compute halting probabilities and masks.

        Args:
            h: Hidden states of shape (B, L, D)
            threshold: Optional override for halting threshold

        Returns:
            probs: Halting probabilities (B, L)
            mask: Boolean mask (B, L) where True indicates halting
            stats: Dictionary with 'act_prob_mean' and 'act_open_rate'
        """
        # h: (B, L, D) -> logits: (B, L, 1) -> squeeze -> (B, L)
        logits = self.proj(h).squeeze(-1)
        probs = mx.sigmoid(logits)

        # Apply threshold
        th = self.threshold if threshold is None else threshold
        mask = probs > th

        # Compute statistics
        stats = {
            "act_prob_mean": mx.mean(probs),
            "act_open_rate": mx.mean(mask.astype(mx.float32)),
        }

        return probs, mask, stats
