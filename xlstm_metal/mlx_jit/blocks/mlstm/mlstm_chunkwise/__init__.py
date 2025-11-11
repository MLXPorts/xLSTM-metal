"""mLSTM Chunkwise Implementation - Clean NCPS Architecture.

Modular mLSTM with clean separation of concerns:
- Projection Cell: Input transformations (Q/K/V, gates)
- Kernel Cells: Pure recurrence logic (parallel vs recurrent)
- Output Cell: Output transformations (norm, gate, projection)
- Neuron: Wires cells together with dispatch logic
"""

from .mlstm_projection_cell import mLSTMProjectionCell
from .mlstm_parallel_kernel_cell import mLSTMParallelKernelCell
from .mlstm_recurrent_kernel_cell import mLSTMRecurrentKernelCell
from .mlstm_output_cell import mLSTMOutputCell
from .mlstm_neuron import mLSTMNeuron

__all__ = [
    'mLSTMProjectionCell',
    'mLSTMParallelKernelCell',
    'mLSTMRecurrentKernelCell',
    'mLSTMOutputCell',
    'mLSTMNeuron',
]
