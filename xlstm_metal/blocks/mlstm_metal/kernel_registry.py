#!/usr/bin/env python
"""
Metal Kernel Registry - Singleton for JIT-compiled kernels

Ensures kernels compile exactly ONCE and persist as globals.
Blocks access kernels via this registry - kernels are NOT passed through MAD data flow.
"""

import mlx.core as mx
from typing import Dict, Optional, Callable


class MetalKernelRegistry:
    """
    Singleton registry for compiled Metal kernels.

    Uses lazy compilation: kernels are compiled on first access, not on init.
    This avoids circular imports and ensures kernels persist for process lifetime.
    """

    _instance: Optional['MetalKernelRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._kernels = {}
            cls._instance._compilers = {}
        return cls._instance

    def register_compiler(self, name: str, compiler_fn: Callable):
        """
        Register a kernel compiler function.

        Args:
            name: Kernel identifier
            compiler_fn: Function that returns compiled kernel (called lazily on first get)
        """
        self._compilers[name] = compiler_fn

    def get_kernel(self, name: str):
        """
        Get compiled kernel by name (compiles on first access - lazy evaluation).

        Args:
            name: Kernel identifier ('fw_recurrent', 'fw_parallel', etc.)

        Returns:
            Compiled Metal kernel function

        Raises:
            KeyError: If kernel compiler not registered
        """
        # Lazy compilation: compile on first access
        if name not in self._kernels:
            if name not in self._compilers:
                raise KeyError(f"Kernel '{name}' not registered. Available: {list(self._compilers.keys())}")

            # Compile kernel (happens once on first access)
            self._kernels[name] = self._compilers[name]()

        return self._kernels[name]


# Global singleton instance
_KERNEL_REGISTRY = MetalKernelRegistry()


def register_kernel(name: str, compiler_fn: Callable):
    """Register a kernel compiler (called at module level in kernel source files)."""
    _KERNEL_REGISTRY.register_compiler(name, compiler_fn)


def get_kernel(name: str):
    """
    Get compiled kernel from global registry.

    Usage:
        from xlstm_metal.blocks.mlstm_metal.kernel_registry import get_kernel
        fw_kernel = get_kernel('fw_recurrent')
        outputs = fw_kernel(inputs=..., grid=..., threadgroup=...)
    """
    return _KERNEL_REGISTRY.get_kernel(name)


__all__ = ['MetalKernelRegistry', 'register_kernel', 'get_kernel']
