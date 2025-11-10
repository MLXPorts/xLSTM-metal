"""
FFN Block - MLX Backend
"""

from .block import xLSTMFeedForwardBlock, FFNConfig
from .gated_ffn_cell import GatedFFNCell
from .gated_ffn import GatedFFN

# Alias for compatibility with existing code
# Note: GatedFFN is now the NCPS-style wrapper, not an alias
# xLSTMFeedForwardBlock is the original block implementation

__all__ = [
    'xLSTMFeedForwardBlock',
    'GatedFFN',
    'GatedFFNCell',
    'FFNConfig',
]
