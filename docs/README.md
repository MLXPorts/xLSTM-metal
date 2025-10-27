# xLSTM-Metal Documentation

Complete technical documentation for the xLSTM-Metal implementation optimized for Apple Silicon.

## Quick Start

New to xLSTM-Metal? Start here:

1. **[Main README](../README.md)** - Installation and basic usage
2. **[AGENTS.md](../AGENTS.md)** - Developer guide and best practices
3. **[MLX Architecture](porting/mlx_metal/XLSTM_MLX_ARCHITECTURE.md)** - Core architecture overview

## Documentation Structure

### Core Architecture

**Primary Implementation:**
- **[MLX Architecture](porting/mlx_metal/XLSTM_MLX_ARCHITECTURE.md)** - Complete system architecture, MAD wiring, kernel design
- **[MLX Inference](porting/mlx_metal/MLX_INFERENCE_ARCHITECTURE.md)** - Inference pipeline and generation
- **[MAD Wiring Integration](components/mad/MAD_WIRING_INTEGRATION.md)** - MAD system design
- **[mLSTM MAD Architecture](components/mad/MAD_MLSTM_ARCHITECTURE.md)** - mLSTM block implementation

### Implementation Guides

**MLX/Metal (Primary):**
- **[MLX Guide](porting/mlx_metal/mlx_guide.md)** - Complete MLX implementation guide
- **[MLX Numerics and DType Guide](porting/mlx_metal/MLX_NUMERICS_AND_DTYPE_GUIDE.md)** - Numerical stability and precision
- **[MLX Runtime Config](porting/mlx_metal/MLX_RUNTIME_CONFIG.md)** - Runtime configuration options
- **[MLX Tuning Guide](porting/mlx_metal/MLX_TUNING_GUIDE.md)** - Performance optimization

**Kernel Development:**
- **[MLX Kernel Patterns](components/kernels/MLX_KERNEL_PATTERNS.md)** - Common kernel patterns
- **[MLX Metal Shader Integration](components/kernels/MLX_METAL_SHADER_INTEGRATION.md)** - Metal shader development
- **[MLX Metal Kernel Guide](components/kernels/MLX_Metal_Kernel_Guide.md)** - Kernel implementation guide

### Testing and Validation

- **[MLX Testing](porting/mlx_metal/MLX_TESTING.md)** - Test suite and validation
- **Run Tests:** `python run_pytest.py`

### Reference Implementations

**Canonical xLSTM:**
- **[Canonical Implementation Notes](porting/CANONICAL_XLSTM_IMPLEMENTATION_NOTES.md)** - Reference implementation details

**PyTorch Reference (for comparison):**
- **[PyTorch MPS Inference](porting/pytorch_mps/PYTORCH_MPS_INFERENCE_ARCHITECTURE.md)** - PyTorch MPS implementation
- **[PyTorch MLX Conv1D Differences](porting/pytorch_mps/PYTORCH_MLX_CONV1D_DIFFERENCES.md)** - Format differences

### Research and Future Work

- **[Extended Context Plan](plan/EXTENDED_CONTEXT_PLAN.md)** - Long context support
- **[Research Notes](plan/RESEARCH_NOTES.md)** - Research directions
- **[Reversible RNN Notes](plan/REVERSIBLE_RNN_NOTES.md)** - Memory-efficient architectures
- **[xLSTM vs LFM2 Comparison](plan/xLSTM_vs_LFM2_Comparison.md)** - Model comparisons

### Advanced Topics

- **[CoreML/ANE Guidance](porting/coreml/ane_guidance.md)** - Apple Neural Engine deployment
- **[Triton Kernels Deep Dive](components/kernels/TRITON_KERNELS_DEEP_DIVE.md)** - Triton kernel reference
- **[MLX vs Ray](porting/mlx_metal/MLX_VS_RAY.md)** - Architecture comparisons

## Key Concepts

### Metal Accelerated Dispatch (MAD)

The MAD system is xLSTM-Metal's core innovation for Apple Silicon:

- **Wired Block System**: Declarative block composition with automatic data flow
- **Backend Agnostic**: Same topology works across MLX, PyTorch, JAX
- **Optimized Scheduling**: Stage-based execution with minimal overhead
- **Type Safety**: Compile-time shape validation

See [MAD Wiring Integration](components/mad/MAD_WIRING_INTEGRATION.md) for details.

### MLX Framework

MLX is Apple's machine learning framework optimized for Apple Silicon:

- **Unified Memory**: Efficient use of shared memory architecture
- **Lazy Evaluation**: Computation graphs evaluated on-demand
- **Metal Backend**: Direct GPU acceleration
- **NumPy-like API**: Familiar interface for array operations

See [MLX Guide](porting/mlx_metal/mlx_guide.md) for implementation patterns.

### xLSTM Architecture

Extended LSTM combines mLSTM (matrix LSTM) and sLSTM (scalar LSTM):

- **mLSTM**: Matrix-valued hidden states with covariance updates
- **sLSTM**: Scalar memory with exponential gating
- **Soft-capped Attention**: Numerical stability for large models
- **Residual Connections**: Skip connections for gradient flow

See [xLSTM-7B Architecture](porting/mlx_metal/XLSTM_MLX_ARCHITECTURE.md) for complete details.

## Usage Patterns

### Basic Inference

```python
from xlstm_metal import xLSTMRunner

# Load model (transformers-like API)
runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

# Generate text
prompt_ids = [1, 2, 3]  # Your tokenized prompt
output_ids = runner.generate(prompt_ids, max_tokens=50)
```

### Command Line

```bash
# Single generation
python generate.py --model NX-AI/xLSTM-7b --prompt "Hello world"

# Interactive mode
python generate.py --model NX-AI/xLSTM-7b --interactive

# Model info
python generate.py --model NX-AI/xLSTM-7b --info
```

See [Main README](../README.md) for more examples.

## Development Workflow

1. **Read [AGENTS.md](../AGENTS.md)** for development guidelines
2. **Check existing implementations** before creating new code
3. **Run tests** with `python run_pytest.py`
4. **Update documentation** for any user-facing changes
5. **Follow MLX patterns** from [MLX Guide](porting/mlx_metal/mlx_guide.md)

## Contributing

When contributing to documentation:

1. Keep examples runnable and tested
2. Include code snippets with actual paths
3. Reference related docs for cross-referencing
4. Update this index when adding new docs
5. Follow the existing structure and style

## Getting Help

- **Issues**: Report bugs on GitHub
- **Documentation**: Browse this docs folder
- **Examples**: See `lab/` for experimental code
- **Tests**: Check `tests/` for usage examples

---

**Note**: This documentation describes the MLX-first implementation. PyTorch references are provided for comparison but are not the primary implementation path.
