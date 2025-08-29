import torch
import torch.nn as nn
from typing import Optional, Dict

from xlstm_official_full.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .cube_gated_block import CubeGatedBlock
from .scheduler import boundary_commit_mask
from .act_halting import ACTHaltingHead
from .telemetry import energy


class HRMXLSTM(nn.Module):
    """xLSTM BlockStack wrapped with HRM+ features (cube gating, ACT, Z5 scheduler).

    - Runs the official xLSTMBlockStack.
    - Applies a CubeGatedBlock at the stack boundary (post-blocks norm output).
    - Updates the Cube only on Z5 boundary steps (slot==4).
    - Emits telemetry: alpha_mean, conf_mean, act_prob_mean, act_open_rate, energy_pre_gate, energy_post_gate.
    """

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.stack = xLSTMBlockStack(config=config)
        d_model = config.embedding_dim
        self.cube_gate = CubeGatedBlock(d_in=d_model)
        self.act_head = ACTHaltingHead(d_model, threshold=0.5)

    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None, **kwargs) -> tuple[torch.Tensor, Dict[str, float]]:
        B, L, D = x.shape
        dev = x.device
        if times is None:
            times = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)

        h = self.stack(x, **kwargs)
        # teacher = identity at boundary in this simple wrapper
        y_teacher = h
        commit_mask = boundary_commit_mask(times)
        y_cg, alpha_mean, conf_mean = self.cube_gate(h, y_teacher=y_teacher, train=self.training, allow_commit=commit_mask)

        # Halting & energy telemetry
        probs, mask, stats = self.act_head(y_cg)
        e_pre = energy(h)
        e_post = energy(y_cg)

        telem = {
            "alpha_mean": alpha_mean,
            "conf_mean": conf_mean,
            "act_prob_mean": stats["act_prob_mean"],
            "act_open_rate": stats["act_open_rate"],
            "energy_pre_gate": float(e_pre.item()),
            "energy_post_gate": float(e_post.item()),
        }
        return y_cg, telem
