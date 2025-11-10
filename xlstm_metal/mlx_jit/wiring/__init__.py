"""
Wiring for MLX Backend

NCPS-based wiring system for xLSTM models. The key wiring approach is:

1. AutoWiring - Automatically creates wiring from safetensors structure (RECOMMENDED)
   - Model-agnostic: works with any xLSTM variant
   - Analyzes safetensors to detect block types (mLSTM, sLSTM, attention)
   - Creates appropriate cells for each block

2. Manual wiring patterns (mLSTMWiring, etc.) - For custom architectures

Usage:
    >>> from xlstm_metal.mlx_jit.wiring import create_auto_wiring
    >>> wiring = create_auto_wiring("xlstm_7b_model")
    >>> # Wiring automatically detected 32 mLSTM blocks
"""

from .auto_wiring import AutoWiring, create_auto_wiring, analyze_safetensors_structure
from .mlstm_wiring import mLSTMWiring, mLSTMBlockWiring, xLSTMStackWiring
from .wirings import Wiring, AutoNCP, Random, FullyConnected, NCP

__all__ = [
    # Auto wiring (recommended)
    'AutoWiring',
    'create_auto_wiring',
    'analyze_safetensors_structure',
    
    # Manual wiring patterns
    'mLSTMWiring',
    'mLSTMBlockWiring',
    'xLSTMStackWiring',
    
    # Base wiring classes
    'Wiring',
    'AutoNCP',
    'Random',
    'FullyConnected',
    'NCP',
]
from .auto_wiring import AutoWiring, create_auto_wiring

__all__ = [
    'Wiring',
    'AutoNCP',
    'Random',
    'FullyConnected',
    'NCP',
    'mLSTMWiring',
    'mLSTMBlockWiring',
    'xLSTMStackWiring',
    'AutoWiring',
    'create_auto_wiring',
]
