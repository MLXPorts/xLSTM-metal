"""
MAD Wiring for MLX Backend
"""

from .wiring import WiredMADModel
from .xlstm_7b import create_xlstm_wiring, xLSTMWiring, AutoNCPxLSTMWiring

__all__ = ['WiredMADModel', 'create_xlstm_wiring', 'xLSTMWiring', 'AutoNCPxLSTMWiring']
