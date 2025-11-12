# xLSTM-Metal: High-Performance xLSTM for Apple Silicon

Production-ready xLSTM (Extended LSTM) implementation optimized for Apple Silicon using MLX and Metal acceleration. Features automatic model loading, config-driven architecture with NCPS wiring patterns, and a simple generation API.

**Author**: Sydney Renee (The Solace Project)  
**Email**: sydney@solace.ofharmony.ai  
**License**: Apache 2.0

## Quick Start

```bash
# Install dependencies
pip install mlx transformers tokenizers

# Run inference with local model
python generate.py --model xlstm_7b_model --prompt "The capital of France is" --max-tokens 50

# Interactive mode
python generate.py --model xlstm_7b_model --interactive
```

## Simple API

xLSTM-Metal uses MLX for native Apple Silicon acceleration with automatic configuration loading:

```python
from xlstm_metal.mlx_jit.generate import xLSTMRunner
from xlstm_metal.mlx_jit.tokenizer import TokenizerBlock, TokenizerConfig

# Load model (config-driven, works with any xLSTM size)
runner = xLSTMRunner("xlstm_7b_model")

# Initialize tokenizer
tokenizer_config = TokenizerConfig(model_path="xlstm_7b_model")
tokenizer = TokenizerBlock(tokenizer_config)

# Generate text
prompt_ids = tokenizer.encode("Hello world").tolist()
generated_ids = runner.generate(
    prompt_ids, 
    max_tokens=50,
    temperature=0.8,
    top_p=0.9
)
output = tokenizer.decode(generated_ids)
print(output)
```

**Key Design Principles:**

- **Config-Driven**: Automatically adapts to any xLSTM model size from `config.json`
- **MLX-Native**: Full Apple Silicon optimization with Metal acceleration
- **NCPS Wiring**: Declarative block composition with automatic structure discovery
- **Simple & Direct**: No heavy abstractions, clear data flow
- **Production-Ready**: Stable numerical handling, proper dtype management

## Features

- Apple Silicon Native**: Optimized for M1/M2/M3/M4 with Metal acceleration via MLX
- Config-Driven Architecture**: Automatically loads and adapts to any xLSTM model size
- NCPS Wiring System**: Neural Circuit Policy-inspired wiring for declarative block composition
- Simple API**: Clean generation interface without heavy abstractions
- Numerical Stability**: Proper dtype handling (float32/bfloat16) for stable inference
- Smart Weight Loading**: Supports safetensors with automatic sharding and structure discovery
- Production-Ready**: Resolved NaN issues, validated on xLSTM-7B model

## What's New (v0.3.0)

**November 2024 - Stable Release**

- ✅ **Fixed dtype handling**: Resolved torch_dtype vs autocast_kernel_dtype confusion
- ✅ **NaN elimination**: Resolved numerical instability in mLSTM blocks
- ✅ **NCPS wiring patterns**: Automatic model structure discovery from safetensors
- ✅ **Comprehensive documentation**: Added detailed docstrings throughout codebase
- ✅ **Validated inference**: Tested and working on xLSTM-7B (32 blocks, 4096d)
- ✅ **Parameter propagation**: Fixed dtype flow through block hierarchy
- ✅ **Better error messages**: Actionable guidance for common issues

See [COMPLETE_FIX_SUMMARY.md](docs/COMPLETE_FIX_SUMMARY.md) and [DOCSTRING_ENRICHMENT_SUMMARY.md](docs/DOCSTRING_ENRICHMENT_SUMMARY.md) for full details.

## Architecture

xLSTM-Metal uses a modular, auto-discovery architecture inspired by Neural Circuit Policies (NCPS):

### Core Components

**1. WiredxLSTM Model** (`xlstm_metal/mlx_jit/models/wired_xlstm.py`)
- Top-level model class that assembles complete xLSTM language models
- Automatic structure discovery from safetensors checkpoints
- Builds correct stack of blocks (mLSTM, sLSTM) dynamically based on checkpoint inspection

**2. NCPS Auto-Wiring** (`xlstm_metal/mlx_jit/wiring/auto_wiring.py`)
- Inspired by Neural Circuit Policy wiring patterns for declarative composition
- Introspects `model.safetensors.index.json` to discover architecture
- Detects block types (mLSTM, sLSTM, attention) from weight key patterns
- Creates sequential connectivity automatically (block_0 → block_1 → ... → block_N)
- Provides factory methods for block cell creation

**3. mLSTM/sLSTM Blocks** (`xlstm_metal/mlx_jit/blocks/`)
- **mLSTM (Matrix LSTM)**: Matrix-valued hidden states with covariance update rules
- **sLSTM (Scalar LSTM)**: Traditional scalar memory with exponential gating
- Modular cell design pattern (stateless transformations)
- Optimized Metal kernels for core operations (matmul, elementwise)

**4. Generation Engine** (`xlstm_metal/mlx_jit/generate.py`)
- xLSTMRunner: High-level inference interface
- Stateful generation for efficient autoregressive decoding
- Temperature, top-k, top-p (nucleus) sampling support
- Stop token handling and BOS token insertion

### Model Architecture Flow

```
Input Token IDs [B, S]
  ↓ embedding (token → vector)
Embeddings [B, S, D]
  ↓ blocks[0..N-1]  (mLSTM/sLSTM with residuals + FFN)
Hidden States [B, S, D]
  ↓ out_norm (RMSNorm)
Normalized [B, S, D]
  ↓ lm_head (Linear projection)
Logits [B, S, vocab_size]
  ↓ soft_cap (tanh-based capping for stability)
  ↓ sampling (temperature/top-k/top-p)
Generated Tokens
```

Each block typically follows this pattern:
```
residual = x
x = norm_mlstm(x)
x, state = mlstm_cell(x, state)  # Stateful recurrence
x = x + residual

residual = x
x = norm_ffn(x)
x = ffn(x)  # Feed-forward network
x = x + residual
```

### NCPS Wiring Benefits

The NCPS-inspired wiring system provides:

1. **Zero-config loading**: Works with any xLSTM checkpoint without hardcoding architecture
2. **Model-agnostic**: Single codebase handles 1B, 7B, 13B, etc. variants
3. **Introspectable**: Query block types and structure before instantiation
4. **Version agnostic**: Adapts to checkpoint structure changes automatically
5. **Modular**: Easy to add new block types (attention, sparse MoE, etc.)

### MLX Backend

MLX provides native Apple Silicon optimization:
- **Unified Memory**: Direct GPU access without CPU-GPU data transfers
- **Lazy Evaluation**: Computation graphs evaluated on-demand for efficiency
- **Metal Kernels**: Hardware-accelerated operations on GPU
- **NumPy-like API**: Familiar array programming interface

## Model Support

### Official Models

- **xLSTM-7B** (`NX-AI/xLSTM-7b`): 7 billion parameter model
    - 32 xLSTM blocks with mLSTM and sLSTM layers
    - 4096 embedding dimensions, 32 attention heads
    - Trained on diverse text corpus

### Custom Models

Any xLSTM model with a `config.json` file is supported. The implementation automatically:

- Detects model architecture from configuration
- Creates appropriate NCPS wiring for block execution
- Loads weights from safetensors or NPZ format

## Installation

```bash
# Install MLX (Apple Silicon only)
pip install mlx

# Install tokenizer support
pip install transformers tokenizers

# Clone repository
git clone https://github.com/SolaceHarmony/xLSTM-Metal.git
cd xLSTM-metal

# Download xLSTM-7B model from HuggingFace (~14GB)
# You can use the provided download script or download manually:
python scripts/downloads/download_model.py

# Or download manually from HuggingFace:
# https://huggingface.co/NX-AI/xLSTM-7b

# Run inference
python generate.py --model ./xlstm_7b_model --prompt "Hello world"
```

### Quick Install

```bash
pip install mlx transformers tokenizers
```

**Note**: This implementation requires Apple Silicon (M1/M2/M3/M4) and macOS 13.0+.

## Usage Examples

### Command Line

```bash
# Basic generation
python generate.py --model xlstm_7b_model --prompt "Once upon a time"

# Advanced sampling
python generate.py --model xlstm_7b_model \
  --prompt "The future of AI" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9

# Interactive mode
python generate.py --model xlstm_7b_model --interactive

# Model information
python generate.py --model xlstm_7b_model --info

# Debug wiring structure
python generate.py --model xlstm_7b_model --prompt "Test" --show-wiring
```

### Python API

```python
from xlstm_metal.mlx_jit.generate import xLSTMRunner
from xlstm_metal.mlx_jit.tokenizer import TokenizerBlock, TokenizerConfig

# Initialize runner
runner = xLSTMRunner("xlstm_7b_model")

# Initialize tokenizer
tokenizer_config = TokenizerConfig(model_path="xlstm_7b_model")
tokenizer = TokenizerBlock(tokenizer_config)

# Get model information
info = runner.get_model_info()
print(f"Model: {info['num_blocks']} blocks, {info['embedding_dim']}d")

# Generate with custom parameters
prompt_ids = tokenizer.encode("Hello world").tolist()
generated_ids = runner.generate(
    prompt_ids,
    max_tokens=100,
    temperature=0.7,
    top_k=50
)
output = tokenizer.decode(generated_ids)
print(output)

# Stateful generation (efficient for long sequences)
runner.reset_state()
prompt_ids = tokenizer.encode("Tell me a story").tolist()
current_ids = prompt_ids
for i in range(50):  # Generate 50 tokens
    next_token = runner.generate_next_token(
        mx.array([current_ids], dtype=mx.int64),
        temperature=0.8
    )
    current_ids = [int(next_token)]
    print(tokenizer.decode([int(next_token)]), end='', flush=True)
```

## Performance

xLSTM-Metal leverages Apple Silicon's unified memory architecture and Metal acceleration through MLX:

- **Unified Memory**: Direct GPU access without CPU-GPU data transfers
- **MLX Optimization**: Lazy evaluation and optimized kernel fusion
- **Efficient Execution**: Sequential block processing with minimal overhead
- **Memory Efficient**: Stateful generation reduces recomputation

Performance characteristics depend on model size, sequence length, and hardware generation (M1/M2/M3/M4).

**Typical Performance (xLSTM-7B on M2 Max)**:
- First token latency: ~500ms (includes model loading and Metal shader compilation)
- Subsequent tokens: ~50-100ms per token
- Memory usage: ~14GB (model weights) + ~2-4GB (inference state)

See [docs/](docs/) for detailed architecture documentation and optimization guides.

## Technical Details

### xLSTM Architecture

The Extended Long Short-Term Memory (xLSTM) architecture combines:

- **mLSTM (matrix LSTM)**: Matrix memory and covariance update rule
- **sLSTM (scalar LSTM)**: Scalar memory with exponential gating
- **Residual Connections**: Skip connections for gradient flow
- **Layer Normalization**: Stable training and inference

### Mathematical Foundation

xLSTM extends traditional LSTM with:

- Matrix-valued hidden states for increased expressiveness
- Exponential gating for improved gradient flow
- Soft attention mechanisms within memory cells
- Architectural scaling for billion-parameter models

### Implementation Highlights

- **Config-Driven Architecture**: Automatic model creation from JSON configuration
- **Weight Loading**: Support for safetensors and NPZ formats
- **Memory Efficiency**: Optimized for Apple's unified memory
- **Type Safety**: Full MLX array type support

## Development

### Repository Structure

```
├── xlstm_metal/              # Core implementation
│   └── mlx_jit/              # MLX backend (primary)
│       ├── generate.py       # Generation runner
│       ├── tokenizer/        # Tokenizer wrapper
│       ├── models/           # WiredxLSTM model
│       ├── wiring/           # NCPS auto-wiring
│       ├── blocks/           # mLSTM/sLSTM blocks
│       └── utils/            # Config and weight loading
├── docs/                     # Technical documentation
├── tests/                    # Test suite
├── scripts/                  # Utilities and tools
└── kernel_development/       # Metal kernel experiments
```

### Testing

```bash
# Run test suite
python run_pytest.py

# Test specific components
python -m pytest tests/test_pretrained_inference.py -v

# Test numerical parity
python test_numerical_parity.py
```

### Contributing

This is an independent port to Apple Silicon. Contributions are welcome! Please:

1. Follow the coding style in existing files
2. Add tests for new features
3. Update documentation as needed
4. See [AGENTS.md](AGENTS.md) for development guidelines

## Credits and Attribution

### This Port

**xLSTM-Metal** is an independent port to Apple Silicon with MLX.

- **Author**: Sydney Renee
- **Organization**: The Solace Project
- **Email**: sydney@solace.ofharmony.ai
- **Repository**: https://github.com/SolaceHarmony/xLSTM-Metal

This port includes:
- MLX backend implementation with Metal acceleration
- NCPS-inspired wiring system for automatic structure discovery
- Numerical stability fixes and dtype handling improvements
- Production-ready inference with proper error handling

### Original Research

xLSTM was introduced by Beck et al. (2024):

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter,
S. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.

### Model Weights

Official xLSTM-7B model weights provided by [NX-AI](https://huggingface.co/NX-AI/xLSTM-7b) under Apache 2.0 license.

### Framework

Built on [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework for Apple Silicon.

### Acknowledgments

- NX-AI team for the original xLSTM research and model weights
- Apple MLX team for the excellent framework
- Neural Circuit Policies (NCPS) research for inspiring the wiring system design

## License

Apache License 2.0. See [LICENSE](LICENSE) for full text.

Model weights from NX-AI are also under Apache 2.0.

## Citation

If you use this implementation, please cite the original xLSTM paper:

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

## Documentation

Complete technical documentation available in [docs/](docs/):

- [MLX Architecture Guide](docs/porting/mlx_metal/XLSTM_MLX_ARCHITECTURE.md)
- [NCPS Wiring System](docs/components/mad/MAD_WIRING_INTEGRATION.md)
- [Testing Guide](docs/porting/mlx_metal/MLX_TESTING.md)
- [Developer Guide](AGENTS.md)
- [Complete Fix Summary](docs/COMPLETE_FIX_SUMMARY.md)
- [Docstring Enrichment](docs/DOCSTRING_ENRICHMENT_SUMMARY.md)

## Requirements

- **Hardware**: Apple Silicon (M1/M2/M3/M4)
- **OS**: macOS 13.0 or later
- **Python**: 3.9 or later
- **MLX**: Latest version from pip

---

*This is an unofficial implementation optimized for Apple Silicon. For the original research and reference
implementation, see the xLSTM paper.*