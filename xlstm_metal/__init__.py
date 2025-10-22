"""
MAD (Mechanistic Architecture Design) Framework

Declarative neural architecture composition using NCPS wiring patterns.
Backend-specific implementations in xlstm_metal.wiring.mlx and xlstm_metal.wiring.torch_compiled.
"""

from .wiring import MADWiring, BlockSpec, BlockType, BackendType

__all__ = ['MADWiring', 'BlockSpec', 'BlockType', 'BackendType']
