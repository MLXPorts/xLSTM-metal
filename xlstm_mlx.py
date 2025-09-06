"""Legacy import shim for xlstm_mlx (deprecated).

Prefer: `from implementations.mlx.xlstm_mlx import ...`.
This shim remains for compatibility and will be removed in a future cleanup.
"""
import warnings as _warnings
_warnings.warn(
    "Importing 'xlstm_mlx' from repo root is deprecated; "
    "use 'implementations.mlx.xlstm_mlx' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from implementations.mlx.xlstm_mlx import *  # noqa: F401,F403
