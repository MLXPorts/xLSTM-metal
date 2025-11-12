"""PyTorch RMSNorm module with optional Metal acceleration."""

from __future__ import annotations

import torch
import torch.nn as nn

from .rms_metal import rms_norm as metal_rms_norm


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
        if x.device.type != "mps":
            raise RuntimeError("RMSNormCell requires tensors on MPS for Metal execution")
        x_norm = x.to(torch.float32) if self.force_float32 else x
        w = self.weight
        if w is not None and w.dtype != x_norm.dtype:
            w = w.to(x_norm.dtype)
        out = metal_rms_norm(x_norm, w, self.eps)
        if self.force_float32 and out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out


__all__ = ["RMSNormCell"]
