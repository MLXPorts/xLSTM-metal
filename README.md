# xLSTM Implementation Suite

Start Here: see AGENTS.md for tools and workflows. If you’re using Claude, read CLAUDE.md for strict do/don’t rules.

A comprehensive collection of xLSTM (Extended LSTM) implementations with varying optimization levels, from basic mathematical prototypes to production-ready systems.

## Overview

This repository contains multiple xLSTM implementations progressing from research prototypes to optimized production systems. The implementations maintain mathematical fidelity to the original xLSTM paper while adding practical enhancements for real-world usage.

## Current MPS Backends and Chunkwise Architecture (2025-08)

On Apple Silicon, inference runs with compiled MPS backends rather than handwritten Metal shaders:
- Step kernel `metal`: `torch.compile`d function executing on MPS with stabilized float32 math.
- Sequence kernel `native_sequence__metal`: decode by looping the compiled step.
- Chunkwise (prefill) backends:
  - `chunkwise--queued_compiled_steps`: CPU thread pool queues many small step kernels; all math on GPU.
  - `chunkwise--ray_compiled_steps`: Ray actors (local_mode) coordinate compiled steps.
  - `chunkwise--native_compiled_autograd`: compiled chunkwise comparator.

Unique scheduling: heads are split into bands and sequences into small chunks (default 32). The coordinator enqueues many tiny compiled step kernels to MPS; per‑band order is preserved while overlapping work across bands/chunks.

Quick start (local HF checkpoint):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
python scripts/runners/xlstm_quick.py --prompt "The capital of France is" --new 20
```

Tuning knobs (MPS): `chunk_size`, `heads_per_band`, `workers`, optional `streams`. See `docs/TUNING_GUIDE.md`.

## Optimization Harness

Automated parameter search lives in `scripts/optimize_mps.py`.
- Modes: `--mode ga` (simple genetic algorithm) or `--mode random`
- Backends: `--backend ray` (tune `heads_per_band`, `chunk_size`) or `--backend queued` (tune `workers`, `heads_per_band`, `chunk_size`)
- Outputs: `runs/mps_opt/<backend_timestamp_tag>/` contains:
  - `run_meta.json` (run config), `trials.jsonl` (all trials), `summary.csv` (table), `best.json` (best params+metrics)

Example (GA, ray):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. XLSTM_RAY_LOCAL_MODE=1 \
  python scripts/optimize_mps.py \
    --backend ray \
    --model_path ./xlstm_7b_model \
    --prompt "The capital of France is" \
    --new 64 \
    --mode ga --generations 5 --population 10 --repeats 1
```

After a run you can regenerate and rate outputs:
```bash
# Save outputs for every row in summary.csv
python scripts/save_outputs_for_trials.py \
  --run runs/mps_opt/<run_dir> \
  --model_path ./xlstm_7b_model \
  --prompt-file ./prompts/long_form.txt \
  --new 32 \
  --outputs runs/mps_opt/<run_dir>/outputs

# Score quality/diversity for saved outputs
python scripts/judge_outputs.py \
  --model_path ./xlstm_7b_model \
  --prompt-file ./prompts/long_form.txt \
  --outputs runs/mps_opt/<run_dir>/outputs
```

### Parameter Search Workflow
- 1) Optimize: run `scripts/optimize_mps.py` (GA or random) to explore `chunk_size`, `heads_per_band` (and `workers` for queued). Results stored in `runs/mps_opt/<run>`.
- 2) Save continuations: `scripts/save_outputs_for_trials.py` regenerates greedy continuations for every row in `summary.csv` into `<run>/outputs` with filenames encoding params.
- 3) Judge quality: `scripts/judge_outputs.py` computes avg_logprob, perplexity, and distinct-2/3 for each output; writes `ratings.jsonl` and `ratings.csv`.
- 4) Plot: `scripts/plot_opt_results.py` summarizes speed vs settings (max decode tok/s by chunk_size/heads/workers; prefill vs decode scatter).

Tip: Prefer the Ray backend (default) with `XLSTM_RAY_LOCAL_MODE=1`; queued remains for environments without Ray.

Monitoring: use `scripts/xltop.py` (TUI) or `--mem-log` on the runner; enable the Ray dashboard with `--ray-dashboard`.

Further docs (Apple/MPS)
- `docs/APPLE_MPS_GUIDE.md` — platform guide and knobs
- `docs/PYTORCH_MPS_INFERENCE_ARCHITECTURE.md` — how our compiled MPS inference works
- `docs/PYTORCH_MPS_FUSION_NOTES.md` — pseudo‑kernel fusion and inner tiling
- `docs/TRITON_KERNELS_DEEP_DIVE.md` — survey of installed Triton kernels (CUDA) and mapping to our MPS path
- `docs/ANE_DEPLOYMENT_GUIDE.md` — deploying to ANE via ane_transformers or Executorch→Core ML
- `docs/STATE_EXPANSION_PRECISION.md` — limb‑precision (bf16×L) for stable recurrent state
- `docs/EMBEDDING_DISAMBIGUATION_NOTES.md` — handling “homonyms” and ambiguity via observables
- `docs/RESEARCH_NOTES.md` — consolidated research notes, evidence, and decisions

## Implementation Status

### Core Implementations

| Implementation | Status | Performance | Features |
|----------------|--------|-------------|----------|
| `xlstm_pytorch_enhanced.py` | ✅ Stable | 730 tok/s | Production-ready, comprehensive config |
| `xlstm_pytorch.py` | ✅ Stable | ~400 tok/s | Base implementation |
| `xlstm_mlx.py` | ✅ Stable | ~300 tok/s | Apple Silicon MLX backend |
| `xlstm_metal_optimized.py` | ⚠️ Beta | TBD | MPS optimization, torch.compile |
| `xlstm_chunked_parallel.py` | ⚠️ Beta | TBD | Chunk-based parallelization |
| `xlstm_streaming_inference.py` | ⚠️ Beta | TBD | Real-time streaming |

### Testing and Utilities

- `xlstm_ultimate_benchmark.py` - Comprehensive benchmarking suite
- `run_tests.py` - Test suite for all implementations
- `train_xlstm.py` - Training framework
- `benchmark.py` - Performance testing utilities

## Key Features

### sLSTM Block
- Exponential gating mechanism preventing gradient vanishing
- State normalization for numerical stability
- Memory mixing within heads
- Optional causal convolution

### mLSTM Block
- Matrix memory structure for enhanced capacity
- Covariance update rule
- Fully parallelizable operations
- Skip connections and grouped normalization

## Installation

```bash
pip install -r requirements.txt
```

For MLX (Apple Silicon only):
```bash
pip install mlx
```

## Usage

### MLX Version
```python
from implementations.mlx.xlstm_mlx import create_xlstm_model  # legacy shim: `from xlstm_mlx import ...` also works

model = create_xlstm_model(
    vocab_size=50257,
    num_layers=12,
    signature=(7, 1),  # 7 mLSTM blocks, 1 sLSTM block pattern
    inp_dim=768,
    head_dim=96,
    head_num=8
)

# Forward pass
tokens = mx.random.randint(0, 50257, (batch_size, seq_len))
logits = model(tokens)
```

### PyTorch Version
```python
from implementations.pytorch.xlstm_pytorch import create_xlstm_model  # legacy shim supported

model = create_xlstm_model(
    vocab_size=50257,
    num_layers=12,
    signature=(7, 1),  # 7 mLSTM blocks, 1 sLSTM block pattern
    inp_dim=768,
    head_dim=96,
    head_num=8,
    device='cuda'
)

# Forward pass
tokens = torch.randint(0, 50257, (batch_size, seq_len))
logits = model(tokens)
```

## Testing

Run the test suite to validate both implementations:

```bash
python test_implementations.py
```

## Model Configuration

Key parameters:
- `vocab_size`: Size of the vocabulary
- `num_layers`: Total number of xLSTM blocks
- `signature`: Tuple (num_mLSTM, num_sLSTM) defining the block pattern
- `inp_dim`: Input/embedding dimension
- `head_dim`: Dimension per attention head
- `head_num`: Number of attention heads
- `p_factor`: Projection factors for (mLSTM, sLSTM)
- `ker_size`: Kernel size for causal convolution
- `dropout`: Dropout probability

## Architecture Details

The xLSTM architecture cycles through the specified signature pattern. For example, with `signature=(7, 1)` and `num_layers=16`, the model will have:
- Blocks 0-6: mLSTM
- Block 7: sLSTM
- Blocks 8-14: mLSTM
- Block 15: sLSTM

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [Official xLSTM Repository](https://github.com/NX-AI/xlstm)
- [MLX xLSTM Implementation](https://github.com/abeleinin/mlx-xLSTM)

## License

This implementation is provided for educational and research purposes.

## Repo Structure (tidy lab)

- `scripts/` – organized runners, benchmarks, downloads, debug tools, and experiments
  - `runners/` – run/train/infer entrypoints
  - `benchmarks/` – throughput and latency harnesses
  - `downloads/` – checkpoint acquisition and loaders
  - `debug/` – MPS probes and debugging scripts
  - `checks/` – sanity checks for weights and assets
  - `build/` – build/environment helpers
  - `experiments/` – ad hoc research utilities
- `implementations/` – organized xLSTM reference/experimental implementations
  - `pytorch/` – base/enhanced, streaming, chunked, compile-fixed
  - `metal/` – MPS/Metal-focused variants
  - `mlx/` – MLX variant and helpers (see also `mlx_implementation/`)
- `prompts/` – long-context prompt snippets
- `examples/` – example apps and demos
- `tests/` – test suite and harnesses
# xLSTM on Apple Silicon

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**
