"""
Experimental CfC-style head update and device-telemetry ring for PyTorch/MPS.

This module mirrors key ideas from MetalCoroutinesTest's `NeuromorphicKernel.metal`:
- Exponential gates (i,f,o) with a running normalizer n (subtract-n variant).
- Candidate update g via sigmoid.
- Continuous-time hidden update (CfC):
    h_new = (h_old + neural_clock * (o * sigmoid(c_new))) / (1 + neural_clock * lambda)
- Optional device-side telemetry ring (tiny int32 tensor) updated at tile boundaries.

Notes
- This is an experiment: it does not replace the canonical xLSTM readout or state math.
- All ops are standard ATen so that `torch.compile` can fuse them on MPS.
- Shapes assume a band slice per call: (B, Hband, D) or (B, Hband) for scalar terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CfcConfig:
    alpha: float = 1e-2  # normalizer step
    target_sum: float = 3.0  # target i+f+o
    clamp_logits: Optional[float] = None  # e.g., 30.0 to avoid extreme exp


def _exp_gate(x: torch.Tensor, n: torch.Tensor, clamp_logits: Optional[float]) -> torch.Tensor:
    z = x - n
    if clamp_logits is not None:
        z = torch.clamp(z, -clamp_logits, clamp_logits)
    return torch.exp(z)


def cfc_head_step(
    h_old: torch.Tensor,  # (B, H, D)
    c_old: torch.Tensor,  # (B, H, D)
    n_old: torch.Tensor,  # (B, H, D)
    # Affine terms per gate; caller provides already-computed preacts of shape (B, H, D)
    gate_i_preact: torch.Tensor,
    gate_f_preact: torch.Tensor,
    gate_o_preact: torch.Tensor,
    gate_g_preact: torch.Tensor,
    
    lambda_vec: torch.Tensor,  # (B, H, D) or broadcastable
    gate_mask: Optional[torch.Tensor] = None,  # (B, H, D) 0/1, optional
    lambda_mask: Optional[torch.Tensor] = None,  # (B, H, D) 0/1, optional
    neural_clock: float = 1.0,
    cfg: CfcConfig = CfcConfig(),
    telemetry_ring: Optional[torch.Tensor] = None,  # (R,) int32 on device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One CfC-style head update step. Returns (h_new, c_new, n_new, fwd_ff) where fwd_ff is o*sigmoid(c_new).
    """
    assert h_old.shape == c_old.shape == n_old.shape == gate_i_preact.shape
    device = h_old.device
    dtype = h_old.dtype

    # Gates (subtract-n variant; exponential gates, sigmoid candidate)
    i_t = _exp_gate(gate_i_preact, n_old, cfg.clamp_logits)
    f_t = _exp_gate(gate_f_preact, n_old, cfg.clamp_logits)
    o_t = _exp_gate(gate_o_preact, n_old, cfg.clamp_logits)
    g_t = torch.sigmoid(gate_g_preact)

    # Cell state
    c_new = f_t * c_old + i_t * g_t

    # Feed-forward read (CfC)
    ff = o_t * torch.sigmoid(c_new)

    # Lambda mask
    lam = lambda_vec
    if lambda_mask is not None:
        lam *= lambda_mask

    # Continuous-time smoothing update for hidden
    denom = 1.0 + neural_clock * lam
    h_new = (h_old + neural_clock * ff) / denom

    # Normalizer update (optional gate mask)
    if gate_mask is None:
        sum_gates = i_t + f_t + o_t
        n_new = n_old + cfg.alpha * (sum_gates - cfg.target_sum)
    else:
        sum_gates = i_t + f_t + o_t
        n_new = torch.where(gate_mask.bool(), n_old + cfg.alpha * (sum_gates - cfg.target_sum), n_old)

    # Optional device-side telemetry: bump a small counter if numerics exceed thresholds
    if telemetry_ring is not None:
        try:
            # Count anomalies on this tile
            not_finite = (~torch.isfinite(h_new)).sum() + (~torch.isfinite(c_new)).sum()
            # Use atomic-style emulation: scatter_add into ring[0]
            # (No real atomics in ATen for MPS; this is best-effort and compiled)
            if not_finite.item() > 0:
                telemetry_ring[0] += int(not_finite.item())
        except Exception:
            pass

    # Promote outputs to original dtype
    h_new = h_new.to(dtype)
    c_new = c_new.to(dtype)
    n_new = n_new.to(dtype)
    ff = ff.to(dtype)
    return h_new, c_new, n_new, ff


def make_telemetry_ring(device: torch.device) -> torch.Tensor:
    """Create a small device-side ring/counter tensor (int32)."""
    return torch.zeros(8, dtype=torch.int32, device=device)


def compiled_cfc_head():
    """Return a torch.compile-wrapped version of cfc_head_step if available."""
    fn = cfc_head_step
    try:
        compile = torch.compile  # type: ignore[attr-defined]
        fn = compile(fn)
    except Exception:
        pass
    return fn

