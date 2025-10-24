"""
MAD Inference Runners

Inference runners for MAD-wired models.

Note: xLSTMRunner is now at xlstm_metal.generate for easy access.
Use: from xlstm_metal import xLSTMRunner
"""

from xlstm_metal.inference.generate import xLSTMRunner
from .xlstm_7b_runner import xLSTM7BRunner

__all__ = ['xLSTMRunner', 'xLSTM7BRunner']
