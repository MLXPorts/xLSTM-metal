"""
MLX-specific blocks for xLSTM

All MLX backend implementations including wiring.
"""

from .mlstm import mLSTMBlock
from .mlstm.xlstm_block import xLSTMBlock
from .mlstm.ffn import GatedFFN
from .slstm import sLSTMBlock, sLSTMLayer
from . import wiring

__all__ = [
    'mLSTMBlock',
    'xLSTMBlock',
    'sLSTMBlock',
    'sLSTMLayer',
    'GatedFFN',
    'wiring',
]

