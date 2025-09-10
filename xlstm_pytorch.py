"""Legacy import shim for xlstm_pytorch (deprecated).

Prefer: `from implementations.pytorch.xlstm_pytorch import ...`.
This shim remains for compatibility and will be removed in a future cleanup.
"""
import warnings as _warnings
_warnings.warn(
    "Importing 'xlstm_pytorch' from repo root is deprecated; "
    "use 'implementations.pytorch.xlstm_pytorch' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from src.pytorch.xlstm_pytorch import *  # noqa: F401,F403
