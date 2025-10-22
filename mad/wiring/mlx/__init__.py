"""
MAD Wiring for MLX Backend
"""

from mad.wiring.mlx.wiring import WiredMADModel
from mad.wiring.mlx.xlstm_7b import create_xlstm_7b_wiring

__all__ = ['WiredMADModel', 'create_xlstm_7b_wiring']
