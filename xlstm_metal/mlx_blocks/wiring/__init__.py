"""NCPS wiring for MLX blocks."""

from .core import Wiring
from .xlstm_7b import xLSTMWiring, AutoNCPxLSTMWiring, create_xlstm_wiring

__all__ = [
    'Wiring',
    'xLSTMWiring',
    'AutoNCPxLSTMWiring',
    'create_xlstm_wiring',
]

