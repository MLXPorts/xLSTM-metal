
"""
Runtime configuration for MLX kernels (no environment variables required).

Use these helpers to configure GEMM tiling options, QR dot mode, and IVF
threadgroup sizing programmatically. Kernel modules consult this state when
building Metal kernels.
"""

from __future__ import annotations

from typing import Optional, Dict, Any


_CONFIG: Dict[str, Any] = {
    # GEMM
    "gemm_pad": None,            # bool | None
    "gemm_align_execw": None,    # bool | None
    "gemm_double_buffer": None,  # bool | None
    # QR
    "qr_dot_mode": None,         # "auto" | "simd" | "simple" | None
    # IVF
    "ivf_tpb": None,             # int | None
    # Model-level
    "fast_head": None,           # bool | None
}


def get_runtime_config() -> Dict[str, Any]:
    return _CONFIG


def configure_gemm(*, pad: Optional[bool] = None,
                   align_execw: Optional[bool] = None,
                   double_buffer: Optional[bool] = None) -> None:
    if pad is not None:
        _CONFIG["gemm_pad"] = bool(pad)
    if align_execw is not None:
        _CONFIG["gemm_align_execw"] = bool(align_execw)
    if double_buffer is not None:
        _CONFIG["gemm_double_buffer"] = bool(double_buffer)


def configure_qr(*, dot_mode: Optional[str] = None) -> None:
    if dot_mode is not None:
        mode = str(dot_mode).lower()
        if mode not in ("auto", "simd", "simple"):
            raise ValueError("qr dot_mode must be one of: auto|simd|simple")
        _CONFIG["qr_dot_mode"] = mode


def configure_ivf(*, tpb: Optional[int] = None) -> None:
    if tpb is not None:
        if tpb <= 0:
            raise ValueError("ivf tpb must be positive")
        _CONFIG["ivf_tpb"] = int(tpb)


def configure_model(*, fast_head: Optional[bool] = None) -> None:
    if fast_head is not None:
        _CONFIG["fast_head"] = bool(fast_head)


def reset_runtime_config() -> None:
    """Resets the runtime configuration to its default values."""
    for k in list(_CONFIG.keys()):
        _CONFIG[k] = None


__all__ = [
    "get_runtime_config",
    "configure_gemm",
    "configure_qr",
    "configure_ivf",
    "configure_model",
    "reset_runtime_config",
]
