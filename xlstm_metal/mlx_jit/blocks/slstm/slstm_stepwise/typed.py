"""Type helpers for Metal kernel dispatch."""

import mlx.core as mx

def u32(val: int) -> mx.array:
    """Create uint32 scalar for Metal kernel parameters."""
    return mx.array(val, dtype=mx.uint32)

__all__ = ['u32']
