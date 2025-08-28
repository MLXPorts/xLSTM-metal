from .fw import mlstm_chunkwise__metal_fw

# Export as both autograd and custbw versions
mlstm_chunkwise__metal_autograd = mlstm_chunkwise__metal_fw
mlstm_chunkwise__metal_custbw = mlstm_chunkwise__metal_fw