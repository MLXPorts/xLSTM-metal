"""
MLX-specific blocks for xLSTM

All MLX backend implementations including wiring.
"""

from .blocks import wiring
from xlstm_metal.mlx_native.blocks.mlstm import mLSTMBlock
from xlstm_metal.mlx_native.blocks.ffn import GatedFFN
from xlstm_metal.mlx_native.blocks.mlstm.xlstm_block import xLSTMBlock
from xlstm_metal.mlx_native.blocks.slstm import sLSTMBlock, sLSTMLayer

__all__ = [
    'mLSTMBlock',
    'xLSTMBlock',
    'sLSTMBlock',
    'sLSTMLayer',
    'GatedFFN',
    'wiring',
]

