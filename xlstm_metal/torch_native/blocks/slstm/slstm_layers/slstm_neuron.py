"""sLSTM Neuron - wires together projection, kernel, and output cells.

The neuron is the complete sLSTM layer that wires together:
    Input → Projection Cell → Kernel Cell (stepwise) → Output Cell → Output

The neuron owns the wiring logic and composes the before/during/after cells.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .slstm_projection_cell import sLSTMProjectionCell
from .stepwise.slstm_stepwise_kernel_cell import sLSTMStepwiseKernelCell
from .slstm_output_cell import sLSTMOutputCell


class sLSTMNeuron(nn.Module):
    """
    sLSTM Neuron - complete sLSTM layer with cell wiring.

    Wires together the sLSTM pipeline:
    1. Projection Cell: x → z, i, f, o (with optional conv)
    2. Kernel Cell: z, i, f, o, state → h, new_state (stepwise recurrence)
    3. Output Cell: h → output (group norm + projection)

    The neuron handles sequential processing across timesteps
    and composes the modular cells.

    Args:
        input_size: Input dimension (embedding_dim)
        num_heads: Number of sLSTM heads
        head_dim: Hidden dimension per head
        conv1d_kernel_size: Conv kernel size (0 = disabled, default 4)
        use_bias: Whether to use bias in projections
        eps: Numerical stability epsilon
        gate_soft_cap: Soft cap value for gates (default 15.0)
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

        # === Before Cell: Projections ===
        self.projection_cell = sLSTMProjectionCell(
            input_size=input_size,
            num_heads=num_heads,
            head_dim=head_dim,
            conv1d_kernel_size=conv1d_kernel_size,
            use_bias=use_bias,
            gate_soft_cap=gate_soft_cap,
        )

        # === During Cell: Stepwise kernel ===
        self.kernel_cell = sLSTMStepwiseKernelCell(
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
        )

        # === After Cell: Output processing ===
        self.output_cell = sLSTMOutputCell(
            input_size=input_size,
            num_heads=num_heads,
            head_dim=head_dim,
            use_bias=use_bias,
            eps=eps,
        )

    @property
    def state_size(self) -> Tuple[int, int, int]:
        """Return state dimensions (c, n, m)."""
        return (
            self.num_heads * self.head_dim,  # c: [B, NH, H]
            self.num_heads * self.head_dim,  # n: [B, NH, H]
            self.num_heads  # m: [B, NH]
        )

    @property
    def output_size(self) -> int:
        """Return output dimension."""
        return self.input_size

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through complete sLSTM neuron.

        Processes sequence step-by-step using kernel cell.

        Args:
            x: Input [B, S, input_size]
            state: Optional previous state (c, n, m)
                   c: [B, NH, H] - cell state
                   n: [B, NH, H] - normalizer
                   m: [B, NH] - stabilizer

        Returns:
            output: Output [B, S, input_size]
            new_state: Updated state (c, n, m)
        """
        B, S, _ = x.shape

        # === Before: Project to gates and z ===
        z, i_preact, f_preact, o_preact, x_conv = self.projection_cell(x)
        # z: [B, S, NH, H]
        # i_preact, f_preact, o_preact: [B, S, NH]

        # === During: Apply kernel step-by-step ===
        # Process each timestep sequentially
        outputs = []
        for t in range(S):
            # Extract timestep
            z_t = z[:, t, :, :]  # [B, NH, H]
            i_t = i_preact[:, t, :]  # [B, NH]
            f_t = f_preact[:, t, :]  # [B, NH]
            o_t = o_preact[:, t, :]  # [B, NH]

            # Apply kernel
            h_t, state = self.kernel_cell(z_t, i_t, f_t, o_t, state)
            # h_t: [B, NH, H]

            outputs.append(h_t)

        # Stack outputs: [B, S, NH, H]
        h = torch.stack(outputs, dim=1)

        # === After: Process output ===
        output = self.output_cell(h)

        return output, state

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "conv1d_kernel_size": self.conv1d_kernel_size,
            "eps": self.eps,
            "gate_soft_cap": self.gate_soft_cap,
        }


__all__ = ['sLSTMNeuron']
