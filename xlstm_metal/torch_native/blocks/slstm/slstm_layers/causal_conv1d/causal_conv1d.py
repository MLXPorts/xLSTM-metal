"""PyTorch-native Causal Conv1d module."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metal_ops import causal_conv1d_mixing, causal_conv1d_depthwise


class CausalConv1d(nn.Module):
    """
    Causal Conv1d cell used in sLSTM projection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias
        )
        
        # Causal padding
        self.padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CausalConv1d.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, S].

        Returns:
            torch.Tensor: Output tensor of shape [B, C, S].
        """
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))

        if x.device.type == "mps":
            return self._forward_mps(x)

        return self.conv(x)

    def _forward_mps(self, x: torch.Tensor) -> torch.Tensor:
        B, C, S = x.shape
        x_seq = x.transpose(1, 2).contiguous()  # [B, S, C]
        bias = self.conv.bias

        if self.groups == self.in_channels:
            weight = self.conv.weight.view(self.out_channels, self.kernel_size).contiguous()
            y = causal_conv1d_depthwise(x_seq, weight, bias)
        elif self.groups == 1:
            weight = self.conv.weight.view(self.out_channels, self.in_channels, self.kernel_size).contiguous()
            y = causal_conv1d_mixing(x_seq, weight, bias)
        else:
            raise ValueError("Unsupported groups for MPS causal conv")

        return y.transpose(1, 2)

    def get_config(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "groups": self.groups,
        }


__all__ = ["CausalConv1d"]
