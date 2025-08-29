import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_cube import MemoryCube


class CubeGatedBlock(nn.Module):
    """Attach a memory cube and a gate to a block boundary.

    Given input h_in [B,L,D], we compute key projections, query the cube for residual predictions,
    compute a confidence-weighted gate α, and blend with a teacher output if provided.
    """

    def __init__(self, d_in: int, d_key: int = None, d_val: int = None):
        super().__init__()
        d_key = d_key or d_in
        d_val = d_val or d_in
        self.key_proj = nn.Linear(d_in, d_key)
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
    ):
        B, L, D = h_in.shape
        keys = self.key_proj(h_in).reshape(B * L, -1)
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
