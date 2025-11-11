"""
MLX tuning helpers: per-device tile selection and default kernel modes.

Loads optional JSON params from `configs/mlx_hardware_params.json` with a
schema like:

{
  "metal": {
    "m3": { "gemm_tiles": { "av": "32x8", "atb": "8x32" }, "qr_dot_mode": "simd" },
    "default": { "gemm_tiles": { "av": "16x16", "atb": "16x16" }, "qr_dot_mode": "auto" }
  }
}

Env override still takes precedence in the kernel modules; this module
provides device-aware defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

try:
    import mlx.core.metal as metal
except Exception:
    metal = None  # type: ignore


def _load_params() -> dict:
    cfg_path = Path("configs/mlx_hardware_params.json")
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    # Built-in defaults
    return {
        "metal": {
            "m3": {"gemm_tiles": {"av": "32x8", "atb": "8x32"}, "qr_dot_mode": "simd"},
            "default": {"gemm_tiles": {"av": "16x16", "atb": "16x16"}, "qr_dot_mode": "auto"},
        }
    }


def device_key() -> Tuple[str, str]:
    """

    :return:
    """
    backend = "metal" if metal is not None else "cpu"
    name = ""
    if metal is not None:
        try:
            info = metal.device_info()
            name = str(info.get("device_name", ""))
        except Exception:
            pass
    return backend, name.lower()


def tiles_for_gemm() -> Tuple[Optional[str], Optional[str]]:
    """Return (av, atb) tile strings or (None, None) if not configured."""
    params = _load_params()
    backend, name = device_key()
    devs = params.get(backend, {})
    if not devs:
        return None, None
    key = "default"
    if "m3" in name:
        key = "m3"
    entry = devs.get(key, {})
    tiles = entry.get("gemm_tiles", {})
    av = tiles.get("av")
    atb = tiles.get("atb")
    return av, atb


def qr_dot_mode_default() -> str:
    """Returns the default QR dot mode for the current device.

    This function returns the default QR dot mode for the current device, which is
    used to optimize the performance of the QR kernels.

    Returns:
        The default QR dot mode for the current device.
    """
    params = _load_params()
    backend, name = device_key()
    devs = params.get(backend, {})
    if not devs:
        return "auto"
    key = "default"
    if "m3" in name:
        key = "m3"
    entry = devs.get(key, {})
    mode = entry.get("qr_dot_mode", "auto")
    return str(mode)


__all__ = [
    "tiles_for_gemm",
    "qr_dot_mode_default",
    "device_key",
]
