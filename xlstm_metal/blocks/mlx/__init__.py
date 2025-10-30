"""
MLX-specific blocks for xLSTM

All MLX backend implementations including wiring.
"""

from xlstm_metal.blocks.mlx import wiring
from xlstm_metal.blocks.mlx.mlstm import mLSTMBlock
from xlstm_metal.blocks.mlx.mlstm.ffn import GatedFFN
from xlstm_metal.blocks.mlx.mlstm.xlstm_block import xLSTMBlock
from xlstm_metal.blocks.mlx.slstm import sLSTMBlock, sLSTMLayer

__all__ = [
    'mLSTMBlock',
    'xLSTMBlock',
    'sLSTMBlock',
    'sLSTMLayer',
    'GatedFFN',
    'wiring',
]

