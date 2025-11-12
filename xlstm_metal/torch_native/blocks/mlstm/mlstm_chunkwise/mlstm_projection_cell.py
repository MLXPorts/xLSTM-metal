"""mLSTM Projection Cell - handles all input projections.

This is the "before" cell in the mLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The projection cell transforms raw inputs into Q/K/V representations
and computes gate pre-activations. It contains NO recurrence logic.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from xlstm_metal.torch_native.blocks.soft_cap.softcap import SoftCap


class mLSTMProjectionCell(nn.Module):
    """
    mLSTM Projection Cell - input transformation only.

    Handles all input projections for mLSTM:
    - Q/K/V projections (feature groups)
    - Input/forget gate projections (excitatory/inhibitory)

    No recurrence, no output processing - just projections.

    Args:
        input_size: Input dimension (embedding_dim)
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head
        v_dim_per_head: Value dimension per head
        use_bias: Whether to use bias in Q/K/V projections
    """

    def __init__(
            self,
            input_size: int,
            num_heads: int,
            qk_dim_per_head: int,
            v_dim_per_head: int,
            use_bias: bool = False,
            gate_soft_cap: Optional[float] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.gate_soft_cap = gate_soft_cap
        self.soft_cap = SoftCap()

        qk_dim = num_heads * qk_dim_per_head
        v_dim = num_heads * v_dim_per_head

        # Q/K/V projections (feature groups in NCPS terminology)
        self.q_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.v_proj = nn.Linear(input_size, v_dim, bias=use_bias)

        # Input/forget gates (excitatory/inhibitory neurons)
        # Gates always have bias=True for proper initialization
        self.igate_proj = nn.Linear(input_size, num_heads)
        self.fgate_proj = nn.Linear(input_size, num_heads)

    def forward(
            self,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project inputs to Q/K/V and gate pre-activations.

        Args:
            x: Input [B, S, input_size]

        Returns:
            q: Query [B, NH, S, DH_qk]
            k: Key [B, NH, S, DH_qk]
            v: Value [B, NH, S, DH_v]
            i_preact: Input gate pre-activation [B, NH, S]
            f_preact: Forget gate pre-activation [B, NH, S]
        """
        B, S, _ = x.shape

        # Project to Q/K/V
        q = self.q_proj(x)  # [B, S, NH*DH_qk]
        k = self.k_proj(x)  # [B, S, NH*DH_qk]
        v = self.v_proj(x)  # [B, S, NH*DH_v]

        # Reshape to multi-head: [B, S, NH, DH] -> [B, NH, S, DH]
        q = q.view(B, S, self.num_heads, self.qk_dim_per_head).permute(0, 2, 1, 3)
        k = k.view(B, S, self.num_heads, self.qk_dim_per_head).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.v_dim_per_head).permute(0, 2, 1, 3)

        # Gate pre-activations: [B, S, NH] -> [B, NH, S]
        i_preact = self.igate_proj(x).permute(0, 2, 1)
        f_preact = self.fgate_proj(x).permute(0, 2, 1)

        if self.gate_soft_cap is not None:
            cap_tensor = torch.tensor(self.gate_soft_cap, dtype=i_preact.dtype, device=i_preact.device)
            i_preact = self.soft_cap(i_preact, cap_tensor)
            f_preact = self.soft_cap(f_preact, cap_tensor)

        return q, k, v, i_preact, f_preact


__all__ = ['mLSTMProjectionCell']
