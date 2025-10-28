"""
MLX-specific blocks for xLSTM

All MLX backend implementations including wiring.
"""

from .mlstm import mLSTMCell, xLSTMBlock
from .mlstm.ffn import GatedFFN
from .slstm import sLSTMLayer
from . import wiring

__all__ = [
    'mLSTMCell',
    'xLSTMBlock',
    'sLSTMLayer',
    'GatedFFN',
    'wiring',
]

