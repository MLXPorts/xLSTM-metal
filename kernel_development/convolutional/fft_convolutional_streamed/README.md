# M2-BERT Metal Kernels

Metal kernel implementations copied from M2-BERT-MLX project for reference and integration into sLSTM.

## Source

Copied from `m2-bert-mlx` 

## Files

- **metal_kernels.py**: Core Metal kernels
  - Depthwise 3-tap convolution
  - Complex multiplication (standard and double-double precision)
  - Demonstrates proper mx.fast.metal_kernel usage patterns

- **metal_fft_conv_streamed.py**: FFT-based convolution kernels
  - FFT kernel compilation
  - Complex multiply in frequency domain
  - IFFT with element-wise operations

- **activations_mlx.py**: Activation function kernels

## Key Patterns

1. **Global kernel compilation**: Compile once on first use, reuse thereafter
2. **Proper threading**: Calculate grid/threadgroup sizes using mx.array scalars
3. **Deterministic operations**: Fixed evaluation order for reproducibility
4. **Extended precision**: Double-double arithmetic for critical operations

## Usage for sLSTM

These kernels provide reference implementations for:
- Causal convolution (can adapt depthwise_conv_3tap)
- Gate operations with numerical stability
- Efficient parallel processing patterns
