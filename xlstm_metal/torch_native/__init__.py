"""
PyTorch-native backend for xLSTM with NCPS wiring.

Provides torch implementations of mLSTM blocks and wiring plus optional
Metal-accelerated kernels (to be added) for Apple Silicon via custom
PyTorch extensions. Falls back to pure torch implementations when the
extension or MPS device is unavailable.
"""

from .wiring import AutoWiring, create_auto_wiring
from .models.wired_xlstm import WiredxLSTM

__all__ = [
    'AutoWiring',
    'create_auto_wiring',
    'WiredxLSTM',
]
