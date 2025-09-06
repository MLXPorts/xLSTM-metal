"""Legacy import shim for xlstm_pytorch_inference (deprecated).

Prefer: `from implementations.pytorch.xlstm_pytorch_inference import ...`.
This shim remains for compatibility and will be removed in a future cleanup.
"""
import warnings as _warnings
_warnings.warn(
    "Importing 'xlstm_pytorch_inference' from repo root is deprecated; "
    "use 'implementations.pytorch.xlstm_pytorch_inference' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from implementations.pytorch.xlstm_pytorch_inference import *  # noqa: F401,F403
