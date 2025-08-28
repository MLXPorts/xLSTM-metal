"""
Experimental: Dendritic Comb Codec (DCC) — Biological implementation (tensorized).

Port of the core algorithm from lambda_neuron.dcc_biological into vectorized
PyTorch ops so we can test weight/state encodings and potential event‑style
representations. This is exploratory and not used by default.

Key idea
- Encode x ∈ R as: residue after L attenuation steps (η<1), plus a set of carry
  events whenever x exceeds a threshold τ at any level. Perfect reconstruction
  uses residue/η^L + Σ excess_level/η^level.

This module provides tensorized encode/decode for batched tensors without
Python lists of events. Carries are represented as per‑level excess tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DccConfig:
    threshold: float = 0.2
    depth: int = 12
    eta: float = (0.5) ** 1.5  # Rall's constant ≈ 0.353553


def dcc_encode_tensor(x: torch.Tensor, cfg: DccConfig = DccConfig()) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode tensor x with DCC. Returns (residue, carries) where:
    - residue: tensor with same shape as x (attenuated after depth steps)
    - carries: tensor with shape (*x.shape, depth) storing per‑level excess (>=0),
               signed to match x's sign; zeros where no spike.
    """
    device = x.device
    dtype = x.dtype
    # Work in float32 for stability; return in original dtype
    xf = x.to(torch.float32)
    sign = torch.sign(xf)
    sign[sign == 0] = 1.0
    residue = xf.abs()
    carries = torch.zeros(*xf.shape, cfg.depth, device=device, dtype=torch.float32)

    for level in range(cfg.depth):
        over = residue - cfg.threshold
        spike = torch.clamp(over, min=0.0)
        # Record signed excess
        carries[..., level] = spike * sign
        # Clamp to threshold then attenuate
        residue = torch.minimum(residue, torch.tensor(cfg.threshold, device=device))
        residue = residue * cfg.eta

    residue = residue * sign
    return residue.to(dtype), carries.to(dtype)


def dcc_decode_tensor(residue: torch.Tensor, carries: torch.Tensor, cfg: DccConfig = DccConfig()) -> torch.Tensor:
    """
    Decode tensorized DCC representation back to original x.
    residue: (*shape)   carries: (*shape, depth)
    """
    dtype = residue.dtype
    rf = residue.to(torch.float32)
    cf = carries.to(torch.float32)
    # Base: residue scaled back from depth attenuations
    recon = rf.abs() / (cfg.eta ** cfg.depth)
    # Accumulate carries: sum over levels spike/eta^level with sign from carries
    for level in range(cfg.depth):
        recon = recon + (cf[..., level].abs() / (cfg.eta ** level))
    # Restore sign: carries already hold sign; residue gives sign if no carries
    sign = torch.sign(rf)
    sign[sign == 0] = 1.0
    # Determine overall sign: prefer residue sign unless carries indicate otherwise
    # (For simplicity, use residue sign as global sign; cf stores signed spikes so
    #  magnitudes accumulate correctly.)
    out = recon * sign
    return out.to(dtype)


def dcc_self_test(device: torch.device | None = None) -> bool:
    dev = device or (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    cfg = DccConfig()
    vals = torch.tensor([0.0, 1.0, -1.0, 0.1, -0.1, 3.14159, -2.71828, 1e-6, -1e-6, 0.5, -0.5, 10.0, -10.0], device=dev)
    res, car = dcc_encode_tensor(vals, cfg)
    rec = dcc_decode_tensor(res, car, cfg)
    err = (vals - rec).abs().max().item()
    return err < 1e-5  # float32 tolerance in vectorized form


__all__ = [
    'DccConfig',
    'dcc_encode_tensor',
    'dcc_decode_tensor',
    'dcc_self_test',
]

