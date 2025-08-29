import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .liquid_time_constant import LiquidTimeConstant
from .cube_gated_block import CubeGatedBlock
from .scheduler import boundary_commit_mask
from .act_halting import ACTHaltingHead
from .telemetry import energy


class TransformerLNN(nn.Module):
    """Hybrid Transformer block with liquid step and cube gating.

    This is a minimal scaffold to ground the documentation; not a full training pipeline.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cube_gate = CubeGatedBlock(d_in=hidden_size)
        self.liquid = LiquidTimeConstant(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_size)
        self.act_head = ACTHaltingHead(hidden_size, threshold=0.5)

    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        device = x.device
        if times is None:
            times = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x)
        attn_in = h.transpose(0, 1)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        h1 = self.norm1(h + attn_out.transpose(0, 1) + self.ffn(h))
        # teacher at block boundary (identity teacher for scaffold)
        y_teacher = h1
        # cube gate (residual form against h1's input is approximated by h for scaffold)
        # commit mask: allow updates only on Z5 boundary (slot==4)
        commit_mask = boundary_commit_mask(times)
        y_cg, alpha_mean, conf_mean = self.cube_gate(
            h1, y_teacher=y_teacher, train=self.training, allow_commit=commit_mask, times=times
        )
        # ACT halting telemetry on gate output
        act_probs, act_mask, act_stats = self.act_head(y_cg)
        # energy audits around gate
        e_pre = energy(h1)
        e_post = energy(y_cg)
        # liquid step per token
        state = torch.zeros(B, self.hidden_size, device=device)
        outs = []
        for t in range(L):
            out_t, state = self.liquid(y_cg[:, t], state, times[:, t])
            outs.append(out_t)
        yhid = torch.stack(outs, dim=1)
        yhid = self.norm2(yhid + y_cg)
        y = self.output_proj(yhid)
        telem = {
            "alpha_mean": alpha_mean,
            "conf_mean": conf_mean,
            "act_prob_mean": act_stats["act_prob_mean"],
            "act_open_rate": act_stats["act_open_rate"],
            "energy_pre_gate": float(e_pre.item()),
            "energy_post_gate": float(e_post.item()),
        }
        return y, telem
