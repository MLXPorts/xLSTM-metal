"""
MAD Wiring for MLX Backend
"""

from .wiring import WiredMADModel
from .xlstm_7b import create_xlstm_7b_wiring
from .hrm_wiring import create_hrm_xlstm_7b_wiring

__all__ = ['WiredMADModel', 'create_xlstm_7b_wiring', 'create_hrm_xlstm_7b_wiring']
