"""
MLX-specific blocks for xLSTM

All MLX backend implementations including wiring.
"""

from xlstm_metal.mlx_jit import wiring

# TODO: Re-enable after restructuring
# from xlstm_metal.blocks.mlstm import mLSTMBlock
# from xlstm_metal.blocks.ffn import GatedFFN
# from xlstm_metal.blocks.mlstm import xLSTMBlock
# from xlstm_metal.blocks.slstm import sLSTMBlock, sLSTMLayer

__all__ = [
    # 'mLSTMBlock',
    # 'xLSTMBlock',
    # 'sLSTMBlock',
    # 'sLSTMLayer',
    # 'GatedFFN',
    'wiring',
]

