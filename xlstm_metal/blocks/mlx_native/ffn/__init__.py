"""
FFN Block - MLX Native Backend (no Metal JIT kernels)
"""

from .block import xLSTMFeedForwardBlock

# Alias for compatibility with existing code
GatedFFN = xLSTMFeedForwardBlock

__all__ = [
    'xLSTMFeedForwardBlock',
    'GatedFFN',
]
