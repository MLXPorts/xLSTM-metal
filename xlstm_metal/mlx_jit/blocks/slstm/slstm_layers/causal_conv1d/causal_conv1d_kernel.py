"""Causal Conv1d Cell - NCPS compliant before-cell for sLSTM."""

from __future__ import annotations

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

        scale = self._compute_scale(kernel_size)
        if channel_mixing:
            shape = (channels, channels, kernel_size)
        else:
            shape = (channels, kernel_size)
        rand = mx.random.normal(shape, dtype=mx.float32)
        self.weight = mx.multiply(rand, scale)
        self.bias = mx.zeros((channels,), dtype=mx.float32)

    def reset_parameters(self) -> None:
        if self.weight is None:
            return
        scale = self._compute_scale(self.kernel_size)
        rand = mx.random.normal(self.weight.shape, dtype=mx.float32)
        self.weight = mx.multiply(rand, scale)
        self.bias = mx.zeros_like(self.bias)

    def __call__(self, x: mx.array) -> mx.array:
        if self.kernel_size <= 0 or self.weight is None:
            return x
        if self.channel_mixing:
            return metal_causal_conv1d_mixing(x, self.weight, self.bias)
        return metal_causal_conv1d_depthwise(x, self.weight, self.bias)

    @staticmethod
    def _compute_scale(kernel_size: int) -> mx.array:
        """Return MLX scalar for He-style initialization scale."""
        one_int = mx.array(1, dtype=mx.int32)
        kernel_tensor = mx.array(kernel_size, dtype=mx.int32)
        denom_int = mx.maximum(one_int, kernel_tensor)
        denom = denom_int.astype(mx.float32)
        base = mx.divide(mx.array(1.0, dtype=mx.float32), denom)
        return mx.power(base, mx.array(0.5, dtype=mx.float32))

    def get_config(self) -> dict:
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "channel_mixing": self.channel_mixing,
        }


__all__ = ["CausalConv1dCell"]
