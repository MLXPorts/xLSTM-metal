# implementations/pytorch

PyTorch implementations of xLSTM used for research and comparison.

Files

- `xlstm_pytorch.py` — Base implementation.
- `xlstm_pytorch_enhanced.py` — Production-oriented config, stability features.
- `xlstm_pytorch_inference.py` — Inference-focused.
- `xlstm_streaming_inference.py` — Streaming API and state caching.
- `xlstm_chunked_parallel.py` — Chunk-based parallelization prototype.
- `xlstm_torch_compile_fixed.py` — Compile-related fixes.

When to use

- Baseline comparisons, ablation studies, and feature exploration outside the compiled MPS path.

