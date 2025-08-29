import torch
import torch.nn as nn
from typing import Optional, Dict

from ..temporal.attention import SpiralAttention
from ..architecture.spiral_layer import UKMSpiralLayer
from .liquid_time_constant import LiquidBlock
from .memory_cube import MemoryCube
from .cube_gated_block import CubeGatedBlock


class TransformerLNNHybrid(nn.Module):
    """
    Minimal SolaceCore HRM+ hybrid block for UKM:
    - Input proj → Liquid (fast module) → SpiralAttention (context) → CubeGatedBlock (memory) → Output proj.
    - Memory cube queried per-batch using a pooled key; updated after forward.
    - Exposes simple telemetry for alpha/conf.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        seq_len: int = 128,
        cube_capacity: int = 512,
        blocky_levels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.liquid = LiquidBlock(hidden_dim, hidden_dim, blocky_levels=blocky_levels)
        self.attn = SpiralAttention(hidden_dim)
        # Memory cube + gating
        self.cube = MemoryCube(key_dim=hidden_dim, value_dim=hidden_dim, capacity=cube_capacity)
        self.gate = CubeGatedBlock(hidden_dim, cube=self.cube, hidden=max(64, hidden_dim // 2))
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = self.liquid(h)
        h = self.attn(h)
        h, telem = self.gate(h)
        y = self.output_proj(h)

        if update_memory and self.training:
            # Update cube with pooled key/value per batch
            with torch.no_grad():
                key = h.mean(dim=1)  # (B,D)
                val = h[:, -1, :]    # last-step summary as value (B,D)
                self.cube.update(key, val)

        return {"output": y, "alpha_mean": torch.tensor(telem["alpha_mean"]).to(y), "conf": torch.tensor(telem["conf"]).to(y)}

