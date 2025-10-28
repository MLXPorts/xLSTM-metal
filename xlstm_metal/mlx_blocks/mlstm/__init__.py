"""
mLSTM MLX Implementation for xLSTM-7B

MLX-based mLSTM blocks (high-level operations, runs on CPU/Metal/CUDA).
"""

from .block import mLSTMBlock, mLSTMConfig
from .xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .mlstm_cell import mLSTMCell, mLSTMCellConfig
from .components import MultiHeadLayerNorm, RMSNorm, soft_cap
from .kernel import mlstm_recurrent_step, mlstm_sequential
from .chunkwise_mlx import mlstm_chunkwise_mlx

__all__ = [
    'mLSTMBlock',
    'mLSTMConfig',
    'xLSTMBlock',
    'xLSTMBlockConfig',
    'mLSTMCell',
    'mLSTMCellConfig',
    'MultiHeadLayerNorm',
    'RMSNorm',
    'soft_cap',
    'mlstm_recurrent_step',
    'mlstm_sequential',
    'mlstm_chunkwise_mlx',
]
