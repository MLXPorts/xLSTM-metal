"""PyTorch soft-cap module with optional Metal kernel acceleration."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _metal_soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Call into custom Metal kernel if available, else fall back."""
    if x.device.type == "mps":
        try:
            return torch.ops.xlstm_metal.soft_cap(x, cap)
        except (RuntimeError, AttributeError):
            pass  # fall through to eager implementation
    return cap * torch.tanh(x / cap)


class SoftCapCell(nn.Module):
    """NCPS-style cell that applies the soft-cap non-linearity."""

    def __init__(self, cap_value: Optional[float] = None) -> None:
        super().__init__()
        self.register_buffer(
            "_cap",
            torch.tensor(float(cap_value) if cap_value is not None else 0.0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, cap_value: Optional[float] = None) -> torch.Tensor:
        cap = float(cap_value) if cap_value is not None else float(self._cap.item())
        if cap <= 0.0:
            return x
        return _metal_soft_cap(x, cap)


soft_cap = SoftCapCell()

__all__ = ["SoftCapCell", "soft_cap"]
