"""Legacy import shim for xlstm_pytorch.

Prefer: `from implementations.pytorch.xlstm_pytorch import ...`.
This shim preserves old imports used by scripts/tests.
"""
from implementations.pytorch.xlstm_pytorch import *  # noqa: F401,F403

