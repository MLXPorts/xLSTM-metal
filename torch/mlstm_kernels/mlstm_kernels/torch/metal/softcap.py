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
    here = Path(__file__).resolve()
    root = here
    for _ in range(max_up):
        # New preferred location
        candidate_new = (root / "../../../.." / "kernels/metal/pytorch_ext/mlstm_metal_backend.mm").resolve()
        if candidate_new.exists():
            return candidate_new
        # Legacy location (before relocation)
        candidate_legacy = (root / "../../../.." / "mlstm_metal_kernels/mlstm_metal_backend.mm").resolve()
        if candidate_legacy.exists():
            return candidate_legacy
        # Archived prototypes location
        candidate_arc = (root / "../../../.." / "research_archive/metal_prototypes/kernels_metal/pytorch_ext/mlstm_metal_backend.mm").resolve()
        if candidate_arc.exists():
            return candidate_arc
        candidate_arc2 = (root / "../../../.." / "research_archive/metal_prototypes/mlstm_metal_kernels/mlstm_metal_backend.mm").resolve()
        if candidate_arc2.exists():
            return candidate_arc2
        root = root.parent
    return None

def _load_backend() -> None:
    global _BACKEND
    if _BACKEND is not None:
        return
    mm = _find_backend_source()
    if mm is None:
        raise ImportError("Metal backend source not found (kernels/metal/pytorch_ext/mlstm_metal_backend.mm)")
    from torch.utils.cpp_extension import load
    _BACKEND = load(
        name="mlstm_metal_backend",
        sources=[str(mm)],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )

def _read_shader_source() -> str:
    here = Path(__file__).resolve()
    root = here
    for _ in range(6):
        met = (root / "../../../.." / "kernels/metal/shaders/mlstm_kernels.metal").resolve()
        if met.exists():
            return met.read_text()
        legacy = (root / "../../../.." / "mlstm_metal_kernels/mlstm_kernels.metal").resolve()
        if legacy.exists():
            return legacy.read_text()
        arc = (root / "../../../.." / "research_archive/metal_prototypes/kernels_metal/shaders/mlstm_kernels.metal").resolve()
        if arc.exists():
            return arc.read_text()
        arc2 = (root / "../../../.." / "research_archive/metal_prototypes/mlstm_metal_kernels/mlstm_kernels.metal").resolve()
        if arc2.exists():
            return arc2.read_text()
        root = root.parent
    raise FileNotFoundError("mlstm_kernels.metal not found")

def metal_soft_cap(x: torch.Tensor, cap_value: float) -> torch.Tensor:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available for Metal soft_cap")
    if x.device.type != "mps":
        raise RuntimeError("Input tensor must be on MPS device")
    _load_backend()
    src = _read_shader_source()
    return _BACKEND.metal_soft_cap_with_source(x, float(cap_value), src)
