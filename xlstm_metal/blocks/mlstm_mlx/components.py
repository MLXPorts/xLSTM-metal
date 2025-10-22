#!/usr/bin/env python
"""
MLX Components for xLSTM

Implements RMSNorm, MultiHeadLayerNorm, and soft-cap using MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


def soft_cap(x: mx.array, cap_value: float) -> mx.array:
    """
    Soft capping: cap_value * tanh(x / cap_value)

    Args:
        x: Input tensor
        cap_value: Soft cap threshold (15.0 for gates, 30.0 for logits)

    Returns:
        Soft-capped tensor
    """
    return cap_value * mx.tanh(x / cap_value)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Used for pre-normalization before mLSTM and FFN layers.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = mx.ones((num_features,))
        if use_bias:
            self.bias = mx.zeros((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass

        Args:
            x: Input tensor [..., num_features]

        Returns:
            Normalized tensor
        """
        input_dtype = x.dtype

        # Force float32 for reductions if requested
        if self.force_float32_reductions:
            x = x.astype(mx.float32)

        # Compute RMS: sqrt(mean(x^2))
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x_norm = x * mx.rsqrt(variance + self.eps)

        # Cast back to input dtype
        x_norm = x_norm.astype(input_dtype)

        # Apply learned weight and bias
        if self.use_weight:
            x_norm = self.weight * x_norm
        if self.use_bias:
            x_norm = x_norm + self.bias

        return x_norm


class MultiHeadLayerNorm(nn.Module):
    """
    Multi-Head Layer Normalization

    Normalizes independently per head, not across all dimensions.
    Critical for xLSTM-7B mLSTM blocks.

    Weight shape: [num_heads, head_dim] not [num_heads * head_dim]!
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            # Shape: [num_heads, head_dim] - one weight vector per head
            self.weight = mx.ones((num_heads, head_dim))
        if use_bias:
            self.bias = mx.zeros((num_heads, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass

        Args:
            x: Input tensor [B, S, num_heads, head_dim]

        Returns:
            Normalized tensor [B, S, num_heads, head_dim]
        """
        input_dtype = x.dtype

        # Force float32 for reductions if requested
        if self.force_float32_reductions:
            x = x.astype(mx.float32)

        # Normalize per head: compute mean/var over head_dim dimension
        mean = mx.mean(x, axis=-1, keepdims=True)  # [B, S, num_heads, 1]
        variance = mx.var(x, axis=-1, keepdims=True)  # [B, S, num_heads, 1]

        x_norm = (x - mean) * mx.rsqrt(variance + self.eps)

        # Cast back to input dtype
        x_norm = x_norm.astype(input_dtype)

        # Apply per-head learned weight
        if self.use_weight:
            # Broadcast weight [num_heads, head_dim] to [B, S, num_heads, head_dim]
            x_norm = self.weight * x_norm
        if self.use_bias:
            x_norm = x_norm + self.bias

        return x_norm
