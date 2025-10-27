# xLSTM-Metal: High-Performance xLSTM for Apple Silicon

A production-ready xLSTM (Extended LSTM) implementation optimized for Apple Silicon using MLX and Metal Accelerated Dispatch (MAD). Features a transformers-compatible API for seamless integration.

## Quick Start

```bash
# Install dependencies
pip install mlx

# Run inference with transformers-like API
python generate.py --model NX-AI/xLSTM-7b --prompt "The capital of France is" --max-tokens 50
```

## Transformers-Compatible API

xLSTM-Metal provides a familiar API that mirrors HuggingFace Transformers:

```python
from xlstm_metal import xLSTMRunner

# Load model from HuggingFace Hub (like transformers)
runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

# Generate text
prompt_ids = [1, 2, 3]  # Your tokenized prompt
generated_ids = runner.generate(
    prompt_ids, 
    max_tokens=50,
    temperature=0.8,
    top_p=0.9
)
```

**Comparison with PyTorch Transformers:**
```python
# PyTorch Transformers
model = AutoModelForCausalLM.from_pretrained("NX-AI/xLSTM-7b")

# xLSTM-Metal (same interface)
runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")
```

## Features

- **MLX-First Architecture**: Optimized for Apple Silicon with Metal acceleration
- **Metal Accelerated Dispatch (MAD)**: Advanced block scheduling for maximum performance  
- **Transformers-Compatible**: Drop-in replacement for HuggingFace models
- **Automatic Model Loading**: Downloads models from HuggingFace Hub automatically
- **Production Ready**: Memory management, error handling, and monitoring
- **Config-Driven**: Automatically adapts to any xLSTM model size

## Architecture

xLSTM-Metal uses a unique Metal Accelerated Dispatch (MAD) architecture:

- **Wired MAD Model**: Declarative wiring system for efficient block execution
- **Automatic Configuration**: Loads model architecture from `config.json`
- **MLX Backend**: Native Apple Silicon optimization with unified memory
- **Stateful Generation**: Efficient autoregressive text generation

## Model Support

### Official Models

- **xLSTM-7B** (`NX-AI/xLSTM-7b`): 7 billion parameter model
  - 32 xLSTM blocks with mLSTM and sLSTM layers
  - 4096 embedding dimensions, 32 attention heads
  - Trained on diverse text corpus

### Custom Models

Any xLSTM model with a `config.json` file is supported. The implementation automatically:
- Detects model architecture from configuration
- Creates appropriate wiring for MAD execution
- Loads weights from safetensors or NPZ format

## Installation

```bash
# Install MLX (Apple Silicon only)
pip install mlx

# Clone repository
git clone https://github.com/MLXPorts/xLSTM-metal.git
cd xLSTM-metal

# Run inference
python generate.py --model NX-AI/xLSTM-7b --prompt "Hello world"
```

## Usage Examples

### Command Line

```bash
# Basic generation
python generate.py --model NX-AI/xLSTM-7b --prompt "Once upon a time"

# Advanced sampling
python generate.py --model NX-AI/xLSTM-7b \
  --prompt "The future of AI" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9

# Interactive mode
python generate.py --model NX-AI/xLSTM-7b --interactive

# Model information
python generate.py --model NX-AI/xLSTM-7b --info
```

### Python API

```python
from xlstm_metal import xLSTMRunner

# Initialize runner
runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

# Get model information
info = runner.get_model_info()
print(f"Model: {info['num_blocks']} blocks, {info['embedding_dim']}d")

# Generate with custom parameters
generated = runner.generate(
    prompt_ids=[1, 2, 3],
    max_tokens=100,
    temperature=0.7,
    top_k=50
)

# Stateful generation (efficient for long sequences)
runner.reset_state()
for token_id in prompt_ids:
    next_token = runner.generate_next_token([token_id])
```

## Performance

xLSTM-Metal is optimized for Apple Silicon:

- **Unified Memory**: Efficient use of Apple's unified memory architecture
- **Metal Kernels**: GPU-accelerated computation via MLX
- **MAD Scheduling**: Optimal block execution order for maximum throughput
- **Memory Monitoring**: Built-in memory usage tracking and management

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
├── xlstm_metal/           # Core implementation
│   ├── inference/         # Generation and inference
│   ├── wiring/           # MAD wiring system
│   ├── blocks/           # xLSTM block implementations
│   └── utils/            # Configuration and weight loading
├── docs/                 # Technical documentation
├── tests/                # Test suite
└── scripts/              # Utilities and tools
```

### Testing

```bash
# Run test suite
python run_pytest.py

# Test specific components
python -m pytest tests/test_xlstm.py -v
```

## Credits and Attribution

### Original Research

xLSTM was introduced in:
- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.

### Model Weights

- **xLSTM-7B**: Provided by [NX-AI](https://huggingface.co/NX-AI) under Apache 2.0 license
- **HuggingFace Hub**: Model hosting and distribution

### Implementation

- **MLX Framework**: Apple's machine learning framework for Apple Silicon
- **Metal Performance Shaders**: GPU acceleration on Apple devices
- **Contributors**: See [AUTHORS.md](AUTHORS.md) for detailed contributions

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The xLSTM-7B model weights are provided under the Apache 2.0 license by NX-AI.

## Citation

If you use xLSTM-Metal in your research, please cite:

```bibtex
@misc{xlstm_metal_2024,
  title={xLSTM-Metal: High-Performance xLSTM for Apple Silicon},
  author={MLXPorts Contributors},
  year={2024},
  url={https://github.com/MLXPorts/xLSTM-metal}
}
```

And the original xLSTM paper:

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

## Support

- **Documentation**: See `docs/` directory for technical details
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join discussions in GitHub Discussions

---

**Note**: This implementation requires Apple Silicon (M1/M2/M3/M4) for optimal performance. MLX is specifically designed for Apple's unified memory architecture.