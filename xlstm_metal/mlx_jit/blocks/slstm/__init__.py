"""
sLSTM MLX Implementation for xLSTM

MLX-based sLSTM blocks implementing scalar LSTM from xLSTM paper Appendix A.
"""

from .block import sLSTMBlock, sLSTMConfig, sLSTMLayer
from .components import RMSNorm, soft_cap, MultiHeadLayerNorm
from .kernel import slstm_recurrent_step, slstm_sequential
from .slstm_neuron import sLSTMCell

__all__ = [
    'sLSTMBlock',
    'sLSTMConfig',
    'sLSTMLayer',
    'sLSTMCell',
    'RMSNorm',
    'soft_cap',
    'MultiHeadLayerNorm',
    'slstm_recurrent_step',
    'slstm_sequential',
]
