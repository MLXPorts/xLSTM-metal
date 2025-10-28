"""
xLSTM for Apple Silicon (MLX)

High-performance xLSTM inference on Apple Silicon using MLX and Metal.
Supports any xLSTM model size with automatic config loading from HuggingFace Hub.

Quick start:
    >>> from xlstm_metal import xLSTMRunner
    >>> runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")
    >>> output = runner.generate([1, 2, 3], max_tokens=50)
"""

from xlstm_metal.inference.generate import xLSTMRunner
from xlstm_metal.mlx_blocks.wiring.core import MADWiring, BlockSpec, BlockType, BackendType

__version__ = "0.1.0"

__all__ = [
    'xLSTMRunner',
    'MADWiring',
    'BlockSpec',
    'BlockType',
    'BackendType',
]
