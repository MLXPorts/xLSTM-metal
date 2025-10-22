#!/usr/bin/env python
# Copyright (c) NXAI GmbH and its affiliates 2024
# Sydney Bach, Solace Harmony
# Based on SWAX architecture from Meta FAIR & JKU Linz

"""
Sliding Window Attention block as MAD-style layer.

Implements efficient sliding window softmax attention for SWAX hybrid architectures.
Based on "Short window attention enables long-term memorization" (Cabannes et al., 2025)

Key insights from SWAX paper:
- Shorter windows (128) encourage mLSTM to learn long-term memory
- Longer windows (2048) provide better short-context reasoning
- Stochastic window size during training gives best of both worlds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from xlstm_solace_torch.mad.init import small_init_init_, wang_init_


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention block for SWAX hybrid architectures.

    Implements causal sliding window softmax attention with:
    - RoPE (Rotary Position Embeddings)
    - Multi-head attention
    - Fixed window size for O(w) complexity per token
    - Optional stochastic window size for training

    Args:
        dim: Model dimension
        num_heads: Number of attention heads (default 16, per SWAX paper)
        window_size: Sliding window size (default 2048)
        qkv_bias: Use bias in QKV projections (default True, per SWAX paper)
        out_bias: Use bias in output projection (default True)
        dropout: Dropout rate (default 0.0)
        rope_theta: RoPE frequency base (default 10000, per SWAX paper)
        num_blocks: Total number of blocks for weight init (default 1)
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        window_size: int = 2048,
        qkv_bias: bool = True,
        out_bias: bool = True,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        num_blocks: int = 1,
        **kwargs
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.num_blocks = num_blocks

        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Scale factor for attention
        self.scale = self.head_dim ** -0.5

        # Initialize RoPE frequencies
        self._init_rope()

        # Initialize weights
        self.reset_parameters()

    def _init_rope(self):
        """Initialize RoPE (Rotary Position Embeddings) frequencies."""
        # Compute frequency for each dimension pair
        # freq_i = 1 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to query or key tensor.

        Args:
            x: Tensor of shape [B, num_heads, S, head_dim]
            positions: Position indices [B, S]

        Returns:
            Rotated tensor of same shape
        """
        # Compute frequencies: [S, head_dim/2]
        freqs = torch.einsum('bs,d->bsd', positions.float(), self.inv_freq)

        # Expand to [B, 1, S, head_dim/2] for broadcasting
        freqs = freqs.unsqueeze(1)

        # Split x into even and odd dimensions
        x_even = x[..., 0::2]  # [B, num_heads, S, head_dim/2]
        x_odd = x[..., 1::2]   # [B, num_heads, S, head_dim/2]

        # Apply rotation
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated

    def _sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute sliding window attention.

        Args:
            q: Query tensor [B, num_heads, S, head_dim]
            k: Key tensor [B, num_heads, S, head_dim]
            v: Value tensor [B, num_heads, S, head_dim]
            window_size: Override default window size (for stochastic training)

        Returns:
            Output tensor [B, num_heads, S, head_dim]
        """
        B, num_heads, S, head_dim = q.shape
        window = window_size if window_size is not None else self.window_size

        # Compute attention scores: [B, num_heads, S, S]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window mask
        # For each position i, attend only to positions [max(0, i-w+1), i]
        positions = torch.arange(S, device=q.device)
        causal_mask = positions.view(-1, 1) >= positions.view(1, -1)  # [S, S]
        window_mask = positions.view(-1, 1) - positions.view(1, -1) < window  # [S, S]
        mask = causal_mask & window_mask

        # Apply mask (set masked positions to -inf)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, num_heads, S, head_dim]

        return output

    def forward(
        self,
        x: torch.Tensor,
        window_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, S, D]
            window_size: Override default window size (for stochastic training)

        Returns:
            Output tensor [B, S, D]
        """
        B, S, D = x.shape

        # QKV projection and split
        qkv = self.qkv(x)  # [B, S, 3*D]
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, S, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        # Compute sliding window attention
        attn_output = self._sliding_window_attention(q, k, v, window_size)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, num_heads, head_dim]
        attn_output = attn_output.reshape(B, S, D)

        # Output projection and dropout
        output = self.dropout(self.out_proj(attn_output))

        return output

    def reset_parameters(self):
        """Initialize weights using canonical xLSTM initialization.

        - qkv: small_init (for input projection)
        - out_proj: wang_init (for output projection, scaled by num_blocks)

        This matches the SWAX paper's initialization strategy.
        """
        # QKV projection uses small_init
        small_init_init_(self.qkv.weight, dim=self.dim)

        # Output projection uses wang_init
        wang_init_(self.out_proj.weight, dim=self.dim, num_blocks=self.num_blocks)

        # Biases are zero-initialized by PyTorch default


class SWABlock(nn.Module):
    """SWAX-style Sliding Window Attention block with pre-norm.

    Wrapper around SlidingWindowAttention with LayerNorm for use in
    SWAX hybrid architectures (alternating with mLSTM blocks).

    Args:
        dim: Model dimension
        num_heads: Number of attention heads (default 16)
        window_size: Sliding window size (default 2048)
        dropout: Dropout rate (default 0.0)
        num_blocks: Total number of blocks for weight init (default 1)
        **kwargs: Additional arguments passed to SlidingWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        window_size: int = 2048,
        dropout: float = 0.0,
        num_blocks: int = 1,
        **kwargs
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = SlidingWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            num_blocks=num_blocks,
            **kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        window_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connection.

        Args:
            x: Input tensor [B, S, D]
            window_size: Override default window size (for stochastic training)

        Returns:
            Output tensor [B, S, D]
        """
        # Pre-norm + attention + residual
        return x + self.attn(self.norm(x), window_size=window_size)
