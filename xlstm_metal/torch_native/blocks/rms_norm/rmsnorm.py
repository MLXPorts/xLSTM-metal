"""PyTorch RMSNorm module with optional Metal acceleration."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _metal_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type == "mps":
        try:
            return torch.ops.xlstm_metal.rms_norm(x, weight, eps)
        except (RuntimeError, AttributeError):
            pass
    norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    out = x * norm
    if weight is not None:
        out = out * weight
    return out


class RMSNormCell(nn.Module):
    """Applies RMSNorm across the last dimension."""

    def __init__(
        self,
        dims: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        force_float32_reductions: bool = True,
        param_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.force_float32 = force_float32_reductions
        if use_weight:
            self.weight = nn.Parameter(torch.ones(dims, dtype=param_dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x.to(torch.float32) if self.force_float32 else x
        w = self.weight
        if w is not None and w.dtype != x_norm.dtype:
            w = w.to(x_norm.dtype)
        out = _metal_rmsnorm(x_norm, w, self.eps)
        if self.force_float32 and out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out


__all__ = ["RMSNormCell"]
