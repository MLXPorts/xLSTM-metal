"""
NCPS wiring helpers for the PyTorch backend.

The current system relies on AutoWiring, which inspects safetensors structure
to create the appropriate neuron graph for any xLSTM checkpoint.
"""

from .auto_wiring import (
    AutoWiring,
    analyze_safetensors_structure,
    create_auto_wiring,
)
from .wirings import Wiring

__all__ = [
    'Wiring',
    'AutoWiring',
    'analyze_safetensors_structure',
    'create_auto_wiring',
]
