"""
FFN Block - MLX Backend
"""

from .block import xLSTMFeedForwardBlock, FFNConfig

# Alias for compatibility with existing code
GatedFFN = xLSTMFeedForwardBlock

__all__ = [
    'xLSTMFeedForwardBlock',
    'GatedFFN',
    'FFNConfig',
]
