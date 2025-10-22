#!/usr/bin/env python
# Copyright (c) NXAI GmbH and its affiliates 2024
# Sydney Bach, Solace Harmony
# Based on canonical xLSTM implementation

"""
mLSTM block as MAD-style layer.

Thin wrapper around canonical mLSTMBlock for MAD framework compatibility.
"""

import torch
import torch.nn as nn
import typing as tp

from xlstm_solace_torch.mad.blocks import mLSTMBlock


class MLSTMBlock(nn.Module):
    """mLSTM block for MAD framework.

    This is a compatibility wrapper that delegates to the canonical
    mLSTMBlock implementation in xlstm_solace_torch.mad.blocks.

    Args:
        dim: Model dimension
        num_heads: Number of mLSTM heads (default 4)
        conv_kernel_size: Causal conv1d kernel size (default 4)
        qkv_proj_blocksize: QKV projection block size (default 4)
        proj_factor: Inner dimension expansion factor (default 2.0)
        context_length: Maximum sequence length for causal mask (required)
        bias: Use bias in linear layers (default False)
        dropout: Dropout rate (default 0.0)
        num_blocks: Number of blocks for weight init scaling (default 1)
        **kwargs: Additional arguments (ignored for compatibility)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        conv_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        proj_factor: float = 2.0,
        context_length: int = 2048,  # Default context length
        bias: bool = False,
        dropout: float = 0.0,
        num_blocks: int = 1,
        **kwargs
    ):
        super().__init__()

        # Delegate to canonical mLSTMBlock
        self.block = mLSTMBlock(
            dim=dim,
            num_heads=num_heads,
            proj_factor=proj_factor,
            conv_kernel_size=conv_kernel_size,
            qkv_proj_blocksize=qkv_proj_blocksize,
            bias=bias,
            dropout=dropout,
            context_length=context_length,
            num_blocks=num_blocks,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, S, D]
            **kwargs: Additional arguments

        Returns:
            Output tensor [B, S, D]
        """
        return self.block(x, **kwargs)

    def step(
        self,
        x: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        conv_state: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        """Single-step recurrent forward pass.

        Args:
            x: Input tensor [B, 1, D] (single timestep)
            mlstm_state: Previous mLSTM state (c, n, m)
            conv_state: Previous conv state

        Returns:
            (output [B, 1, D], {"mlstm_state": ..., "conv_state": ...})
        """
        return self.block.step(x, mlstm_state=mlstm_state, conv_state=conv_state)

    def reset_parameters(self):
        """Reset parameters using canonical initialization."""
        self.block.reset_parameters()
