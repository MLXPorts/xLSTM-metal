"""xLSTM Models - NCPS-style wired models for MLX backend."""

from .wired_xlstm import WiredxLSTM
from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_block import mLSTMBlock
from xlstm_metal.mlx_jit.blocks.slstm.slstm_block import sLSTMBlock

# Backwards compatibility aliases
xLSTM7BCell = mLSTMBlock
xLSTMsLSTMCell = sLSTMBlock

__all__ = [
    'WiredxLSTM',
    'mLSTMBlock',
    'sLSTMBlock',
    'xLSTM7BCell',
    'xLSTMsLSTMCell',
]
