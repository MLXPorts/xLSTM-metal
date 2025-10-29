"""Metal soft_cap JIT loader and wrapper.

Loads the ObjC++ Metal backend and exposes a Python function that calls the
Metal-accelerated soft_cap when running on MPS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

_BACKEND = None


def _find_backend_source(max_up: int = 6) -> Optional[Path]:
    """Find the mlstm_metal_backend.mm file."""
    here = Path(__file__).resolve()
    root = here
    for _ in range(max_up):
        # Current location: kernel_development/pytorch_mm
        candidate = (root / "../../pytorch_mm/mlstm_metal_backend.mm").resolve()
        if candidate.exists():
            return candidate
        root = root.parent
    return None


def _find_shader_source(max_up: int = 6) -> Optional[Path]:
    """Find the mlstm_kernels.metal shader file."""
    here = Path(__file__).resolve()
    root = here
    for _ in range(max_up):
        # Current location: kernel_development/pytorch_mm
        candidate = (root / "../../pytorch_mm/mlstm_kernels.metal").resolve()
        if candidate.exists():
            return candidate
        root = root.parent
    return None


def _load_backend() -> None:
    """JIT compile and load the Metal backend."""
    global _BACKEND
    if _BACKEND is not None:
        return

    mm = _find_backend_source()
    if mm is None:
        raise ImportError("Metal backend source not found (mlstm_metal_backend.mm)")

    from torch.utils.cpp_extension import load
    _BACKEND = load(
        name="mlstm_metal_backend",
        sources=[str(mm)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=True,
    )


def _read_shader_source() -> str:
    """Read the Metal shader source code."""
    shader_path = _find_shader_source()
    if shader_path is None:
        raise FileNotFoundError("mlstm_kernels.metal not found")
    return shader_path.read_text()


def metal_soft_cap(x: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Apply soft cap operation using Metal kernel.

    Args:
        x: Input tensor on MPS device
        cap_value: Positive cap value

    Returns:
        Tensor with soft cap applied: cap_value * tanh(x / cap_value)
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available for Metal soft_cap")
    if x.device.type != "mps":
        raise RuntimeError("Input tensor must be on MPS device")

    _load_backend()
    src = _read_shader_source()
    return _BACKEND.metal_soft_cap_with_source(x, float(cap_value), src)
