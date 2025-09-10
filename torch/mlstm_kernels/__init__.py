"""Compatibility shim: torch.mlstm_kernels

Use this namespace to import PyTorch mLSTM kernels. For now, it proxies to
the root package `mlstm_kernels` while we migrate code.
"""

import importlib
import sys

_mod = importlib.import_module("mlstm_kernels")

# Expose attributes
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)

# Help the import system find submodules under this namespace
__path__ = getattr(_mod, "__path__", [])
__spec__ = getattr(_mod, "__spec__", None)

# Also register this module name to point at the underlying package
sys.modules[__name__] = _mod

