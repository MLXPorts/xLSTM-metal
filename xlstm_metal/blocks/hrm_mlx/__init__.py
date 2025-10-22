#!/usr/bin/env python
"""
HRM+ (Hierarchical Retrieval Memory) blocks for MLX.

Ports of the PyTorch HRM+ research components to MLX for integration
with the MAD (Modular Atomically-wired Differentiable) framework.
"""

from .scheduler import z5_slots, boundary_commit_mask
from .memory_cube import MemoryCubeMLX
from .act_halting import ACTHaltingHeadMLX
from .liquid_cell import LiquidTimeConstantMLX
from .cube_gated import CubeGatedBlockMLX
from .hrm_xlstm_block import HRMxLSTMBlockMLX, HRMxLSTMConfig

__all__ = [
    'z5_slots',
    'boundary_commit_mask',
    'MemoryCubeMLX',
    'ACTHaltingHeadMLX',
    'LiquidTimeConstantMLX',
    'CubeGatedBlockMLX',
    'HRMxLSTMBlockMLX',
    'HRMxLSTMConfig',
]
