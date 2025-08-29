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

    def __init__(self, config: xLSTMBlockStackConfig, use_cube: bool = True, use_act: bool = True, fuse_phase_keys: bool = True, k_5ht: float = 0.5):
        super().__init__()
        self.stack = xLSTMBlockStack(config=config)
        d_model = config.embedding_dim
        self.use_cube = use_cube
        self.use_act = use_act
        self.cube_gate = CubeGatedBlock(d_in=d_model, fuse_phase_keys=fuse_phase_keys, k_5ht=k_5ht) if use_cube else None
        self.act_head = ACTHaltingHead(d_model, threshold=0.5) if use_act else None
        self.k_5ht = float(k_5ht)

    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None, mod_5ht: Optional[torch.Tensor] = None, **kwargs) -> tuple[torch.Tensor, Dict[str, float]]:
        B, L, D = x.shape
        dev = x.device
        if times is None:
            times = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)

        h = self.stack(x, **kwargs)
        # Optional serotonin (5-HT) divisive gain applied post-stack
        y_cg = h
        gain = None
        if mod_5ht is not None:
            g = mod_5ht
            if g.dim() == 2:
                g = g.unsqueeze(-1)
            gain = torch.exp(-self.k_5ht * g).clamp(0.3, 1.0)
            y_cg = y_cg * gain
        alpha_mean = conf_mean = 0.0
        if self.use_cube:
            # teacher = identity at boundary in this simple wrapper
            y_teacher = h
            commit_mask = boundary_commit_mask(times)
            y_cg, alpha_mean, conf_mean = self.cube_gate(
                y_cg, y_teacher=y_teacher, train=self.training, allow_commit=commit_mask, times=times, mod_5ht=mod_5ht
            )

        # Halting & energy telemetry
        stats = {"act_prob_mean": 0.0, "act_open_rate": 0.0}
        if self.use_act:
            # Raise halting threshold with higher mean 5-HT (patience)
            th = None
            if mod_5ht is not None:
                th = self.act_head.threshold + 0.1 * float(mod_5ht.mean().item())
            probs, mask, stats = self.act_head(y_cg, threshold=th)
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
        if mod_5ht is not None:
            telem["mod_5ht_mean"] = float(mod_5ht.mean().item())
            telem["gain_5ht_mean"] = float(gain.mean().item()) if gain is not None else 1.0
        return y_cg, telem
