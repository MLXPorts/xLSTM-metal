#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Forward kernels (inference)
from .fw_kernel_parallel import mlstm_chunkwise_parallel_fw_Hintra_metal
from .fw_kernel_recurrent import mlstm_chunkwise_recurrent_fw_C_metal

# Backward kernels (training) - TODO: add when needed
# from .bw_kernel_parallel_dK import mlstm_chunkwise__parallel_bw_dK_kernel
# from .bw_kernel_parallel_dQ import mlstm_chunkwise__parallel_bw_dQ_kernel
# from .bw_kernel_parallel_dV import mlstm_chunkwise__parallel_bw_dV_kernel
# from .bw_kernel_recurrent import mlstm_chunkwise__recurrent_bw_dC_kernel
