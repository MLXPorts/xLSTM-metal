"""
sLSTM MLX Implementation for xLSTM

MLX-based sLSTM blocks implementing scalar LSTM from xLSTM paper Appendix A.
Follows NCPS pattern with modular cells, kernels, and neurons.
"""

from .slstm_projection_cell import sLSTMProjectionCell
from .slstm_stepwise.slstm_stepwise_kernel import sLSTMStepwiseKernelCell
from .slstm_output_cell import sLSTMOutputCell
from .slstm_neuron import sLSTMNeuron

__all__ = [
    'sLSTMProjectionCell',
    'sLSTMStepwiseKernelCell',
    'sLSTMOutputCell',
    'sLSTMNeuron',
]
