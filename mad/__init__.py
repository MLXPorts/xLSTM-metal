"""
MAD (Mechanistic Architecture Design) Framework

Declarative neural architecture composition using NCPS wiring patterns.
Backend-specific implementations in mad.wiring.mlx and mad.wiring.torch_compiled.
"""

from mad.wiring import MADWiring, BlockSpec, BlockType, BackendType

__all__ = ['MADWiring', 'BlockSpec', 'BlockType', 'BackendType']
