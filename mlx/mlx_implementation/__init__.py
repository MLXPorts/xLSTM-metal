"""Compatibility shim: mlx.mlx_implementation -> top-level mlx_implementation (reference)."""
import importlib, sys
_mod = importlib.import_module("mlx_implementation")
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)
__path__ = getattr(_mod, "__path__", [])
__spec__ = getattr(_mod, "__spec__", None)
sys.modules[__name__] = _mod

