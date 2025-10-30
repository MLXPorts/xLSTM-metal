"""
MLX-specific blocks for xLSTM

All MLX backend implementations including wiring.
"""

from . import wiring
from .mlstm import mLSTMBlock
from .ffn import GatedFFN
from .mlstm.xlstm_block import xLSTMBlock
from .slstm import sLSTMBlock, sLSTMLayer

__all__ = [
    'mLSTMBlock',
    'xLSTMBlock',
    'sLSTMBlock',
    'sLSTMLayer',
    'GatedFFN',
    'wiring',
]

