"""
mLSTM MLX Implementation for xLSTM-7B

MLX-based mLSTM blocks (high-level operations, runs on CPU/Metal/CUDA).
"""

from .block import mLSTMBlock, mLSTMConfig
from .components import MultiHeadLayerNorm, RMSNorm, soft_cap
from .kernel import mlstm_recurrent_step, mlstm_sequential
from .chunkwise_mlx import mlstm_chunkwise_mlx
from .xlstm_block import xLSTMBlock, xLSTMBlockConfig

__all__ = [
    'mLSTMBlock',
    'mLSTMConfig',
    'MultiHeadLayerNorm',
    'RMSNorm',
    'soft_cap',
    'mlstm_recurrent_step',
    'mlstm_sequential',
    'mlstm_chunkwise_mlx',
    'xLSTMBlock',
    'xLSTMBlockConfig',
]
