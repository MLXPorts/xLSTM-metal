import torch
import torch.nn as nn
from .memory_cube import MemoryCube


class CubeGatedBlock(nn.Module):
    """
    Residual block that blends features with memory-cube retrievals.

    y = alpha(x, conf) * x + (1 - alpha(x, conf)) * (x + P(mem(q)))
    where q = Q(x) is a learned key; mem(q) is top‑K value; conf is concentration.
    """

    def __init__(self, dim: int, cube: MemoryCube, hidden: int = 256):
        super().__init__()
        self.cube = cube
        # Projections for key/value adaptation
        self.key_proj = nn.Linear(dim, cube.key_dim)
        self.mem_proj = nn.Linear(cube.value_dim, dim)
        # Gate uses feature + confidence scalar
        self.gate = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        q = self.key_proj(x).mean(dim=1)  # global key per batch (cheap + stable)
        mem_val, conf, _, _ = self.cube.query(q)
        mem = self.mem_proj(mem_val).unsqueeze(1).expand(B, L, D)
        alpha = self.gate(torch.cat([x, torch.full((B, L, 1), conf, device=x.device, dtype=x.dtype)], dim=-1))
        y = alpha * x + (1 - alpha) * (x + mem)
        return self.norm(y), {"alpha_mean": float(alpha.mean().item()), "conf": conf}
