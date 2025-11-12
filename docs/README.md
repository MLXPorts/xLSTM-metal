# xLSTM-Metal Technical Documentation

**Author:** Sydney Renee (The Solace Project)  
**Contact:** sydney@solace.ofharmony.ai  
**License:** Apache 2.0

Production MLX implementation of xLSTM for Apple Silicon. This documentation describes the working system as of January 2025.

## What This Is

An independently developed port of NX-AI's xLSTM-7B to Apple Silicon using MLX and Metal acceleration. The implementation leverages techniques and patterns from my prior work:

- **NCPS Auto-Wiring** - Adapted from [ncps-mlx](https://github.com/MLXPorts/ncps-mlx) for declarative model composition
- **Metal Kernel Patterns** - Informed by [m2-bert-mlx](https://github.com/MLXPorts/m2-bert-mlx) numerical stability work
- **Precision Enforcement** - Built on [ember-ml](https://github.com/SolaceHarmony/ember-ml) linting infrastructure
- **Unified Memory Optimization** - Techniques shared with [Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx)

## Quick Navigation

### Core Documentation
- **[Architecture Overview](ARCHITECTURE.md)** - System design and component hierarchy
- **[Numerical Stability](NUMERICAL_STABILITY.md)** - Lessons learned from debugging NaN issues
- **[Metal Kernels](METAL_KERNELS.md)** - Custom kernel implementation and optimization
- **[NCPS Wiring](NCPS_WIRING.md)** - Auto-discovery and model composition

### Implementation Guides
- **[Getting Started](GETTING_STARTED.md)** - Installation and first inference
- **[API Reference](API_REFERENCE.md)** - Python API documentation
- **[Performance Tuning](PERFORMANCE.md)** - Optimization strategies for M1/M2/M3/M4

### Development
- **[Contributing](CONTRIBUTING.md)** - Development guidelines and workflow
- **[Testing](TESTING.md)** - Test suite and validation procedures
- **[Debugging](DEBUGGING.md)** - Common issues and solutions

## What Works (January 2025)

**✅ Stable Inference**
- xLSTM-7B generation on M1/M2/M3/M4 Mac
- Float32 precision with proper dtype handling
- 32-block sequential execution via NCPS auto-wiring
- Configurable temperature, top-k, top-p sampling

**✅ Metal Acceleration**
- Custom kernels for mLSTM chunkwise operations
- Optimized RMSNorm and SoftCap implementations
- Unified memory architecture leverage

**✅ Production Tooling**
- Config-driven model loading
- SafeTensors weight handling
- Comprehensive error messages
- Telemetry and profiling hooks

## What Doesn't Work Yet

**⚠️ Known Limitations**
- **Long Context**: Current implementation tested to ~2K tokens, extended context (>16K) experiments archived
- **Mixed Precision**: bfloat16 causes numerical drift; staying with float32 for stability
- **Parallel Execution**: Blocks execute sequentially; multi-head parallelization experiments shelved
- **sLSTM Integration**: Only mLSTM blocks implemented; sLSTM and Liquid Time-Constant experiments in archive

See `docs_archive/` for experimental work on these topics.

## Project Structure

```
xLSTM-metal/
├── xlstm_metal/
│   └── mlx_jit/              # Production MLX implementation
│       ├── generate.py       # Inference runner
│       ├── models/           # WiredxLSTM model
│       ├── wiring/           # NCPS auto-wiring
│       ├── blocks/           # mLSTM, FFN, normalization
│       ├── utils/            # Config, weights, tokenizer
│       └── tokenizer/        # Tokenizer integration
├── docs/                     # Current documentation
├── docs_archive/             # Historical experiments and learnings
├── tests/                    # Test suite
├── tools/                    # emberlint, embercoach, profiling
└── quarantine/               # Experimental/broken code
```

## The Journey (Lessons Learned)

This port required solving several non-obvious problems. Key learnings documented:

1. **Dtype Confusion**: xLSTM config has `torch_dtype`, `autocast_kernel_dtype`, and `inference_state_dtype` - using the wrong one causes NaNs. See [docs_archive/COMPLETE_FIX_SUMMARY.md](../docs_archive/COMPLETE_FIX_SUMMARY.md).

2. **FFT Normalization**: PyTorch and MLX handle `irfft` normalization differently. One applies `1/n`, the other doesn't. Getting this wrong causes order-of-magnitude errors. See [docs_archive/NUMERIC_STABILITY_TORCH_vs_MLX.md](../docs_archive/NUMERIC_STABILITY_TORCH_vs_MLX.md).

3. **Python Scalars**: Using `float()`, `int()`, `.item()` in hot paths breaks graph optimization and introduces double-rounding. Zero-tolerance policy enforced via emberlint. See [docs_archive/NUMERIC_STABILITY_TORCH_vs_MLX.md](../docs_archive/NUMERIC_STABILITY_TORCH_vs_MLX.md).

4. **Metal Shader Limits**: Metal has stricter limits than CUDA (threadgroup memory, argument counts). Kernels need different tiling strategies. See [docs_archive/FIX_RMSNORM_METAL_KERNEL.md](../docs_archive/FIX_RMSNORM_METAL_KERNEL.md).

5. **State Management**: mLSTM has (C, n, m) state tensors with specific shapes and update rules. Getting covariance dimensions wrong (k⊗v vs v⊗k) silently produces garbage. See [docs_archive/architecture/MLSTM_NUMERICAL_STABILITY_ANALYSIS.md](../docs_archive/architecture/MLSTM_NUMERICAL_STABILITY_ANALYSIS.md).

## Experimental Work (Archived)

The `docs_archive/` directory contains experiments and architectural explorations:

- **Extended Context** ([plan/EXTENDED_CONTEXT_PLAN.md](../docs_archive/plan/EXTENDED_CONTEXT_PLAN.md)) - Hierarchical prefill, working memory pools, dynamic allocation strategies
- **State Expansion** ([plan/STATE_EXPANSION_PRECISION.md](../docs_archive/plan/STATE_EXPANSION_PRECISION.md)) - Double-bf16 limb representations for bandwidth reduction
- **Reversible RNNs** ([plan/REVERSIBLE_RNN_NOTES.md](../docs_archive/plan/REVERSIBLE_RNN_NOTES.md)) - Memory-efficient backprop strategies
- **MAD Wiring** ([components/mad/MAD_WIRING_INTEGRATION.md](../docs_archive/components/mad/MAD_WIRING_INTEGRATION.md)) - Original stage-based composition (replaced by NCPS)
- **LFM2 Comparison** ([plan/xLSTM_vs_LFM2_Comparison.md](../docs_archive/plan/xLSTM_vs_LFM2_Comparison.md)) - Architecture analysis

These represent real research and development effort but are not part of the current production system.

## Attribution

**Original Research:**
- xLSTM architecture by Beck et al. (NX-AI)
- arXiv:2405.04517 and arXiv:2503.13427
- Model weights: [NX-AI/xLSTM-7b](https://huggingface.co/NX-AI/xLSTM-7b)

**This Port:**
- Sydney Renee (The Solace Project)
- Independent implementation for Apple Silicon
- Not affiliated with NX-AI

**Related Work:**
- [ncps-mlx](https://github.com/MLXPorts/ncps-mlx) - NCPS wiring patterns for MLX
- [m2-bert-mlx](https://github.com/MLXPorts/m2-bert-mlx) - Metal kernel techniques
- [ember-ml](https://github.com/SolaceHarmony/ember-ml) - Precision enforcement tooling
- [Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx) - Unified memory optimization

## Getting Help

1. **Documentation**: Start with [ARCHITECTURE.md](ARCHITECTURE.md) and [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Issues**: File bugs on GitHub with `generate.py` output and system info
3. **Development**: Read [AGENTS.md](../AGENTS.md) for coding guidelines
4. **Historical Context**: Check `docs_archive/` for past experiments

## Contributing

Contributions welcome. Follow these principles:

1. **Zero-Mock Policy**: No fake implementations in production paths
2. **Reuse First**: Search codebase before implementing
3. **Precision First**: Run emberlint before committing
4. **Document Trials**: If experiments fail, document why in `docs_archive/`
5. **Test Everything**: Run `python run_pytest.py` before PRs

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

**This is working, production-tested code.** The docs reflect what actually runs, not aspirational features. Experimental work is clearly marked and archived separately.
