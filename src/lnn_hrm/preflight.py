import os
import sys


def has_mps() -> bool:
    try:
        import torch
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


def has_ray() -> bool:
    try:
        import ray  # noqa: F401
        return True
    except Exception:
        return False


def assert_mps(msg: str | None = None):
    if not has_mps():
        raise RuntimeError(msg or "MPS is required (Apple unified memory); no CPU/CUDA fallback configured.")


def assert_ray(msg: str | None = None):
    if os.getenv("HRM_SKIP_RAY_CHECK", "0") == "1":
        return
    if not has_ray():
        raise RuntimeError(msg or "Ray is required for xLSTM compiled/scaled operation.")


def preflight_or_exit():
    try:
        assert_mps()
        assert_ray()
    except Exception as e:
        print(f"[preflight] {e}", file=sys.stderr)
        sys.exit(1)

