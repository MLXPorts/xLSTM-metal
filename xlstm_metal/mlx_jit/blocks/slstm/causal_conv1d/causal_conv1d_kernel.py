"""Causal Conv1d Cell - NCPS compliant before-cell for sLSTM."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .depthwise_kernel import metal_causal_conv1d_depthwise
from .mixing_kernel import metal_causal_conv1d_mixing

class CausalConv1dCell(nn.Module):
    """Causal Conv1d cell used in sLSTM projection (NCPS before-cell).

    Args:
        channels: Feature dimension (must equal embedding dim)
        kernel_size: Temporal kernel size (0 disables conv)
        channel_mixing: If True, uses full mixing (Conv1d groups=1). Otherwise
            depthwise per-channel kernels (canonical default).
    """

    def __init__(self, channels: int, kernel_size: int, channel_mixing: bool = False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.channel_mixing = channel_mixing

        if kernel_size <= 0:
            self.weight = None
            self.bias = None
            return

        scale = (1.0 / max(1, kernel_size)) ** 0.5
        if channel_mixing:
            shape = (channels, channels, kernel_size)
        else:
            shape = (channels, kernel_size)
        self.weight = mx.random.normal(shape, dtype=mx.float32) * scale
        self.bias = mx.zeros((channels,), dtype=mx.float32)

    def reset_parameters(self) -> None:
        if self.weight is None:
            return
        scale = (1.0 / max(1, self.kernel_size)) ** 0.5
        self.weight = mx.random.normal(self.weight.shape, dtype=mx.float32) * scale
        self.bias = mx.zeros_like(self.bias)

    def __call__(self, x: mx.array) -> mx.array:
        if self.kernel_size <= 0 or self.weight is None:
            return x
        if self.channel_mixing:
            return metal_causal_conv1d_mixing(x, self.weight, self.bias)
        return metal_causal_conv1d_depthwise(x, self.weight, self.bias)

    def get_config(self) -> dict:
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "channel_mixing": self.channel_mixing,
        }


__all__ = ["CausalConv1dCell"]
