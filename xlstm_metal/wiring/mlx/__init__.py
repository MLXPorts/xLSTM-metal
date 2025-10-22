"""
MAD Wiring for MLX Backend
"""

from .wiring import WiredMADModel
from .xlstm_7b import create_xlstm_7b_wiring

__all__ = ['WiredMADModel', 'create_xlstm_7b_wiring']
