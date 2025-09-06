"""Legacy import shim for xlstm_metal_complete (deprecated).

Prefer: `from implementations.metal.xlstm_metal_complete import ...`.
This shim remains for compatibility and will be removed in a future cleanup.
"""
import warnings as _warnings
_warnings.warn(
    "Importing 'xlstm_metal_complete' from repo root is deprecated; "
    "use 'implementations.metal.xlstm_metal_complete' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from implementations.metal.xlstm_metal_complete import *  # noqa: F401,F403
