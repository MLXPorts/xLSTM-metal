"""
sLSTM Components - Shared utilities

Reuses components from mLSTM where applicable.
"""

import mlx.core as mx
import mlx.nn as nn

# Import shared components from mLSTM
from xlstm_metal.mlx_jit.blocks.mlstm.multihead_norm.multihead_norm import (
    MultiHeadLayerNorm,
    MultiHeadRMSNorm
)

def soft_cap(values: mx.array, cap_value: float) -> mx.array:
    """Apply soft capping using tanh."""
    if cap_value is None:
        return values
    cap = mx.array(cap_value, dtype=values.dtype)
    return mx.multiply(cap, mx.tanh(mx.divide(values, cap)))

# For compatibility
RMSNorm = nn.RMSNorm

__all__ = [
    'RMSNorm',
    'soft_cap',
    'MultiHeadLayerNorm',
    'MultiHeadRMSNorm',
]
