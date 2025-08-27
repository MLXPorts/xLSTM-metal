# Metal Kernel Guide for xLSTM

This is the MLX Metal kernel guide, copied from the ember-ml project for reference in implementing proper Metal kernels for xLSTM operations.

## Key Concepts for xLSTM Metal Kernels:

1. **Use @mx.custom_function decorator** for clean interfaces
2. **Metal kernels provide function body only** - MLX generates signatures
3. **Input naming**: `inp0`, `inp1`, etc. with `inp0_shape[0]` for dimensions
4. **Output naming**: `out0`, `out1`, etc.
5. **Thread positioning**: `thread_position_in_grid.x` for global position
6. **Synchronization**: `threadgroup_barrier(mem_flags::mem_device)` for dependencies
7. **Numerical stability**: Guard against division by zero, NaN propagation

## xLSTM-Specific Metal Optimizations:

- **Soft capping**: `cap_value * tanh(x / cap_value)` 
- **Matrix memory updates**: Outer products for mLSTM
- **Sequential processing**: For recurrent dependencies
- **Causal convolution**: Only past/current time steps
- **RMSNorm**: Parallel variance computation with reduction

The Metal kernels in `xlstm_metal_kernels.py` implement these patterns for optimal xLSTM performance on Apple Silicon.