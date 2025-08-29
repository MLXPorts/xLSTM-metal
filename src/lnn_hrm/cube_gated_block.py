import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_cube import MemoryCube


class CubeGatedBlock(nn.Module):
    """Attach a memory cube and a gate to a block boundary.

    Given input h_in [B,L,D], we compute key projections, query the cube for residual predictions,
    compute a confidence-weighted gate α, and blend with a teacher output if provided.
    """

    def __init__(self, d_in: int, d_key: int = None, d_val: int = None, fuse_phase_keys: bool = True):
        super().__init__()
        d_key = d_key or d_in
        d_val = d_val or d_in
        self.fuse_phase_keys = fuse_phase_keys
        self.key_proj = nn.Linear(d_in, d_key)
        # Optional phase fusion: map [key || phase] -> key_dim
        self.phase_proj = nn.Linear(d_key + 8, d_key)
        self.alpha_head = nn.Sequential(
            nn.Linear(d_in + d_val + 1, d_in), nn.SiLU(), nn.Linear(d_in, 1)
        )
        self.cube = MemoryCube(d_key=d_key, d_val=d_val)
        self.ln_in = nn.LayerNorm(d_in)
        self.ln_pred = nn.LayerNorm(d_val)

    def forward(
        self,
        h_in: torch.Tensor,
        y_teacher: torch.Tensor = None,
        train: bool = False,
        allow_commit: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
    ):
        B, L, D = h_in.shape
        keys = self.key_proj(h_in)
        if self.fuse_phase_keys and times is not None:
            # Build lightweight phase features per token (fast/mid/slow + Z5 slot one-hot)
            # times: (B,L)
            t = times.float()
            # three cos phases with periods 1,3,9 relative units
            phi = torch.stack([
                torch.cos(2 * torch.pi * t / 1.0),
                torch.cos(2 * torch.pi * t / 3.0),
                torch.cos(2 * torch.pi * t / 9.0),
                torch.sin(2 * torch.pi * t / 1.0),
                torch.sin(2 * torch.pi * t / 3.0),
                torch.sin(2 * torch.pi * t / 9.0),
            ], dim=-1)
            # Z5 slot one-hot (5)
            slot = (t.long() % 5)
            z5 = torch.nn.functional.one_hot(slot, num_classes=5).float()
            # concat and reduce to 8 dims (6 trig + 5 one-hot -> 11 -> project to 8 via linear on channel dim)
            # For simplicity, pick first 8 features: 3 cos + 3 sin + first 2 of one-hot
            # but better: a small linear on the 11-dim feature to 8 dims per position
            phase_full = torch.cat([phi, z5], dim=-1)
            phase_feats = torch.tanh(torch.nn.functional.linear(phase_full, torch.eye(phase_full.size(-1), device=h_in.device)[:8]))
            # Broadcast phase to key dim via concat and linear
            k_cat = torch.cat([keys, phase_feats], dim=-1)
            keys = self.phase_proj(k_cat)

        keys = keys.reshape(B * L, -1)
        pred, conf = self.cube.query(keys)
        pred = pred.view(B, L, -1)
        conf = conf.view(B, L, 1)
        feats = torch.cat([self.ln_in(h_in), self.ln_pred(pred), conf], dim=-1)
        alpha = torch.sigmoid(self.alpha_head(feats)).clamp(0, 1)
        y_resid = h_in + pred
        if y_teacher is None:
            y_out = (1 - alpha) * h_in + alpha * y_resid
        else:
            y_out = (1 - alpha) * y_teacher + alpha * y_resid
        if train and y_teacher is not None:
            delta = (y_teacher.detach() - h_in.detach()).reshape(B * L, -1)
            if allow_commit is not None:
                # only update for allowed positions
                mask = allow_commit.reshape(B * L)
                if mask.any():
                    self.cube.update(keys[mask], delta[mask])
            else:
                self.cube.update(keys, delta)
        return y_out, alpha.mean().item(), conf.mean().item()
