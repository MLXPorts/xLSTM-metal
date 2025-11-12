import torch
import torch.nn as nn


class MultiHeadLayerNorm(nn.Module):
    """Multi-head LayerNorm flattening heads after per-head normalization.

    Input: [B, S, NH, DH] -> Normalize over DH per head -> flatten to [B, S, NH*DH].
    Weight/bias are flat of length NH*DH (matches HF xLSTM style).
    """
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, use_weight: bool = True, use_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_heads * head_dim))
        else:
            self.register_parameter('weight', None)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_heads * head_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, NH, DH = x.shape
        if NH != self.num_heads or DH != self.head_dim:
            raise ValueError(f"Expected shape [B,S,{self.num_heads},{self.head_dim}], got {x.shape}")
        # Compute mean/var per head
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_flat = x_norm.view(B, S, -1)
        if self.weight is not None:
            x_flat = x_flat * self.weight
        if self.bias is not None:
            x_flat = x_flat + self.bias
        return x_flat


class MultiHeadRMSNorm(nn.Module):
    """Multi-head RMSNorm flatten heads after per-head RMS normalization."""
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, use_weight: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_heads * head_dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, NH, DH = x.shape
        if NH != self.num_heads or DH != self.head_dim:
            raise ValueError(f"Expected shape [B,S,{self.num_heads},{self.head_dim}], got {x.shape}")
        rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x / rms).type_as(x)
        x_flat = x_norm.view(B, S, -1)
        if self.weight is not None:
            x_flat = x_flat * self.weight
        return x_flat


__all__ = ["MultiHeadLayerNorm", "MultiHeadRMSNorm"]
