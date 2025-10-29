"""
PyTorch Backend for xLSTM

Provides PyTorch implementations of xLSTM blocks using native kernels.
"""

from .block import mLSTMLayer, mLSTMConfig, LayerNorm, soft_cap

__all__ = [
    'mLSTMLayer',
    'mLSTMConfig',
    'LayerNorm',
    'soft_cap',
]
