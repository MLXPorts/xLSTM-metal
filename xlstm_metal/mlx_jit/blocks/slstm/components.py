"""
sLSTM Components - Shared utilities

Reuses components from mLSTM where applicable (RMSNorm, soft_cap).
"""

# Import shared components from mLSTM
from xlstm_metal.blocks.mlstm import RMSNorm, soft_cap, MultiHeadLayerNorm

__all__ = [
    'RMSNorm',
    'soft_cap',
    'MultiHeadLayerNorm',
]
