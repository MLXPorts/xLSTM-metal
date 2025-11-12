"""sLSTM Cell (Scalar LSTM) – Torch Implementation (single-step)

Overview
--------
The scalar LSTM (sLSTM) layer is the *scalar memory* variant from the xLSTM
architecture (see Appendix A of the xLSTM paper). Unlike matrix-memory
(mLSTM), each head maintains a scalar (per-feature) exponential moving
state that enables long-range memory with stable gating.

Design Principles
-----------------
NCPS Pattern: All trainable parameters (projections, norms, optional
conv pre-processing, output projection) are contained in the cell.
The cell processes a SINGLE timestep – batched over the batch dimension –
allowing step-wise autoregressive inference while reusing the same code path.

Computation Flow (per timestep t)
---------------------------------
1. Optional causal Conv1d provides local temporal mixing for gates i,f.
2. Linear projections produce:
   - z_t  : candidate content (reshaped to [B, NH, H])
   - i_t  : input gate preactivation (per head)
   - f_t  : forget gate preactivation (per head)
   - o_t  : output gate preactivation (per head)
3. Soft-cap (tanh-based) applied to gate preactivations (configurable).
4. Exponential stabilization (log-space trick):
   f_log = -softplus(-f_t) gives log(sigmoid(f_t)) without overflow.
   m_t   = max(f_log + m_{t-1}, i_t) keeps denominator stable.
5. Normalized exponential gates:
   f_gate = exp(f_log + m_{t-1} - m_t)
   i_gate = exp(i_t             - m_t)
6. State updates:
   c_t = f_gate * c_{t-1} + i_gate * z_t          (content accumulator)
   n_t = f_gate * n_{t-1} + i_gate                (normalizer accumulator)
   m_t retained as stabilized scalar head memory.
7. Hidden construction:
   h_tilde = c_t / (n_t + eps)
   h       = sigmoid(o_t) * h_tilde
8. Per-head LayerNorm (flatten heads) then output projection to input_size.

Shapes
------
Input:  inputs            [B, D]
States: c_state           [B, NH, H]
        n_state           [B, NH, H]
        m_state           [B, NH]
Output: output            [B, D]
New hx: (c_new, n_new, m_new)

Arguments
---------
input_size          : Embedding / feature dimension D
num_heads           : Number of scalar heads (NH)
head_dim            : Hidden dimension per head (H); NH * H == hidden_size
conv1d_kernel_size  : >0 enables causal conv preprocessing for i,f gates
use_bias            : Bias on projections (disabled for gate norms by default)
eps                 : Numerical stability constant for division
gate_soft_cap       : Value for tanh soft-capping of gate preactivations

Soft-Cap Function
-----------------
cap * tanh(x / cap)     (identity when cap is None).
Prevents extremely large magnitudes that could destabilize exponentials.

Autograd Considerations
-----------------------
All operations are differentiable; the exponential stabilization avoids
large exponent/underflow regions improving gradient stability for long sequences.

This implementation intentionally mirrors the PyTorch reference logic while
using pure PyTorch ops for portability. Where performance-critical,
custom Metal kernels (MPS) can replace projected operations via a torch extension.
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

from xlstm_metal.torch_native.blocks.mlstm.multihead_norm.multihead_norm import MultiHeadLayerNorm


class sLSTMCell(nn.Module):
    """Scalar LSTM (sLSTM) single-step recurrence cell.

    Implements one autoregressive timestep with stabilized exponential gates
    and per-head normalization. ALL parameters reside inside the cell (NCPS style).

    Forward Signature:
        forward(inputs, hx=None) -> (output, new_state)

    Parameters
    ----------
    input_size : int
        Feature / embedding dimension D.
    num_heads : int
        Number of scalar heads (NH).
    head_dim : int
        Per-head hidden dimension H.
    conv1d_kernel_size : int, default 4
        Enables optional causal Conv1d preprocessing for i,f gates when >0.
    use_bias : bool, default False
        Whether linear projections include bias.
    eps : float, default 1e-6
        Numerical stability term in normalization division.
    gate_soft_cap : float, default 15.0
        Soft-cap value; None disables capping.

    Returns
    -------
    output : torch.Tensor [B, D]
        Projected hidden representation for this timestep.
    new_hx : tuple (c_new, n_new, m_new)
        Updated states for next timestep.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        head_dim: int,
        conv1d_kernel_size: int = 4,
        use_bias: bool = False,
        eps: float = 1e-6,
        gate_soft_cap: float = 15.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv1d_kernel_size = conv1d_kernel_size
        self.eps = eps
        self.gate_soft_cap = gate_soft_cap
        hidden_size = num_heads * head_dim

        if conv1d_kernel_size > 0:
            self.causal_pad = conv1d_kernel_size - 1
            self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=conv1d_kernel_size)
            self.conv_act = nn.SiLU()
        else:
            self.conv1d = None

        self.z_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.igate_proj = nn.Linear(input_size, num_heads)
        self.fgate_proj = nn.Linear(input_size, num_heads)
        self.ogate_proj = nn.Linear(input_size, num_heads)

        self.group_norm = MultiHeadLayerNorm(num_heads=num_heads, head_dim=head_dim, eps=eps)
        self.out_proj = nn.Linear(hidden_size, input_size, bias=use_bias)

    @property
    def state_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def output_size(self) -> int:
        return self.input_size

    def soft_cap_gates(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_soft_cap is None:
            return x
        cap = torch.tensor(self.gate_soft_cap, dtype=x.dtype, device=x.device)
        return cap * torch.tanh(x / cap)

    def forward(
        self,
        inputs: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ts: Optional[float | torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process a single timestep.

        Parameters
        ----------
        inputs : torch.Tensor [B, D]
            Current timestep features.
        hx : Optional tuple (c, n, m)
            Previous state; if None initializes zeros.
        ts : Optional[float | torch.Tensor]
            Timestep placeholder (unused, kept for NCPS API compatibility).

        Returns
        -------
        output : torch.Tensor [B, D]
            Projected hidden after normalization & gating.
        new_hx : tuple
            Updated (c_state, n_state, m_state) for next step.

        Notes
        -----
        - Uses stabilized log-sigmoid via softplus for forget gate.
        - Avoids direct sigmoid for f gate accumulation to reduce drift.
        - Maintains scalar normalization (n_state) to keep c_state bounded.
        """
        B = inputs.shape[0]
        NH = self.num_heads
        H = self.head_dim

        if self.conv1d is not None:
            x_seq = inputs.unsqueeze(1)
            if self.causal_pad > 0:
                padding = torch.zeros((B, self.causal_pad, self.input_size), dtype=inputs.dtype, device=inputs.device)
                x_seq = torch.cat([padding, x_seq], dim=1)
            x_conv = self.conv_act(self.conv1d(x_seq))
            x_conv = x_conv[:, -1, :]
        else:
            x_conv = inputs

        z_t = self.z_proj(inputs)
        i_t = self.igate_proj(x_conv)
        f_t = self.fgate_proj(x_conv)
        o_t = self.ogate_proj(inputs)

        i_t = self.soft_cap_gates(i_t)
        f_t = self.soft_cap_gates(f_t)
        o_t = self.soft_cap_gates(o_t)

        z = z_t.view(B, NH, H)

        if hx is None:
            c_state = torch.zeros((B, NH, H), dtype=inputs.dtype, device=inputs.device)
            n_state = torch.zeros((B, NH, H), dtype=inputs.dtype, device=inputs.device)
            m_state = torch.zeros((B, NH), dtype=inputs.dtype, device=inputs.device)
        else:
            c_state, n_state, m_state = hx

        f_log = -torch.nn.functional.softplus(-f_t)
        m_new = torch.maximum(f_log + m_state, i_t)
        i_gate = torch.exp(i_t - m_new)
        f_gate = torch.exp(f_log + m_state - m_new)
        o_gate = torch.sigmoid(o_t)

        i_expanded = i_gate.unsqueeze(-1)
        f_expanded = f_gate.unsqueeze(-1)
        o_expanded = o_gate.unsqueeze(-1)

        c_new = f_expanded * c_state + i_expanded * z
        n_new = f_expanded * n_state + i_expanded
        h_tilde = c_new / (n_new + torch.tensor(self.eps, dtype=inputs.dtype, device=inputs.device))
        h = o_expanded * h_tilde

        h_norm_input = h.unsqueeze(1)  # [B,1,NH,H]
        h_norm = self.group_norm(h_norm_input)  # [B,1,NH*H]
        h_norm = h_norm.squeeze(1)
        output = self.out_proj(h_norm)

        new_hx = (c_new, n_new, m_new)
        return output, new_hx


__all__ = ["sLSTMCell"]
