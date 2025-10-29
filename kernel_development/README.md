# Experimental Kernels: Research and Development Archive

This directory contains kernel prototypes, research experiments, and alternative implementations developed during the xLSTM-Metal optimization process. These are **research artifacts** preserved for reference, learning, and potential future use.

## Purpose

Experimental kernels serve multiple purposes:
- **Historical Record**: Document optimization attempts and design decisions
- **Educational Resource**: Provide reference implementations for learning Metal/MLX
- **Benchmarking Baseline**: Compare production kernels against alternative approaches
- **Future Research**: Foundation for extended context, quantization, or new architectures

⚠️ **Note**: For production use, refer to `xlstm_metal/blocks/` and `xlstm_metal/wiring/`. Experimental kernels are not optimized for production workloads.

## Directory Structure

```
experimental_kernels/
├── README.md                          # This file
├── kernel_guide.md                    # Comprehensive Metal kernel development guide
├── MLX_IMPLEMENTATION_GUIDE.md        # MLX-specific implementation patterns
├── GEMM_TILING_ATTEMPTS.md           # GEMM tiling research and learnings
│
├── mlx_fast_kernels/                  # Optimized MLX kernel prototypes
│   ├── gemm_av.py                    # GEMM A·V kernel
│   ├── gemm_at_b.py                  # GEMM Aᵀ·B kernel
│   └── matmul_tiled.py               # Tiled matrix multiplication
│
├── 2025-09-10-mlx-research/          # September 2025 research sprint
│   ├── README.md                     # Research journal and findings
│   ├── kernels/
│   │   └── mhln_kernels.py          # Multi-head LayerNorm SIMD kernel
│   ├── gemm_tile_bench.py           # GEMM tiling benchmarks
│   ├── mlx_softcap_bench.py         # Soft-cap operation benchmarks
│   ├── mlx_sequence_precompute_scan_demo.py  # Sequence precompute patterns
│   └── mx_streams_overlap_demo.py    # MLX stream overlap experiments
│
├── metal_prototypes/                  # Raw Metal shader prototypes
│   └── (Direct Metal Shading Language experiments)
│
├── torch_msltm_kernels/              # PyTorch MPS kernel experiments
│   └── utils/benchmark/              # Benchmarking utilities
│
├── docs/                             # Detailed research reports
│   ├── MLX_MHLN_Kernel_Report.md    # Multi-head LayerNorm analysis
│   └── 2025-09-10-mlx-research-journal.md  # Research journal
│
└── test_gemm_kernels.py              # GEMM kernel validation tests
```

## Key Research Areas

### 1. Multi-Head LayerNorm (MHLN) Kernels

**Location**: `2025-09-10-mlx-research/kernels/mhln_kernels.py`  
**Report**: `docs/MLX_MHLN_Kernel_Report.md`

**Approach**: SIMD-optimized per-head normalization using one simdgroup (32 threads) per (batch, num_heads) row.

**Key Findings**:
- Achieves 2-4x speedup over MLX ops for moderate head dimensions (96-192)
- Numeric parity with max error ~1e-6
- Uses threadgroup memory with two-barrier reduction pattern
- Best performance at DH >= 96 on Apple M3 Ultra

**Integration Status**: Prototype - can be integrated under feature flag for moderate head dimensions

### 2. GEMM Tiling Strategies

**Location**: `GEMM_TILING_ATTEMPTS.md`, `mlx_fast_kernels/`

**Approaches Explored**:
- **Naive 1D Launch**: Simple but memory-bound (baseline)
- **2D Tile-based**: Block-level tiling with shared memory
- **Multi-kernel Split**: Separate `gemm_av` and `gemm_at_b` kernels
- **Fused Covariance**: Single-pass Z = Aᵀ(A·V) computation

**Key Learnings**:
- Two-kernel approach (`gemm_av` → `gemm_at_b`) outperforms fused for large matrices
- Tile size 16×16 provides good balance on Apple Silicon
- Non-square tiles beneficial for rectangular matrices
- MLX kernel fusion often competitive with custom Metal for small operations

**Status**: Multi-kernel approach adopted in production; fused variants archived

### 3. Soft-Cap Operations

**Location**: `2025-09-10-mlx-research/mlx_softcap_bench.py`

**Implementation**: `cap * tanh(x / cap)` for numerical stability in large models

**Benchmarks**: Compared MLX elementwise ops vs custom Metal kernel

**Finding**: MLX's optimized elementwise ops are competitive; custom kernel overhead not justified for this simple operation

**Status**: Using MLX native ops in production (`xlstm_metal/blocks/mlstm_mlx/components.py`)

### 4. Sequence Precompute and Streaming

**Location**: `2025-09-10-mlx-research/mlx_sequence_precompute_scan_demo.py`, `mx_streams_overlap_demo.py`

**Research Questions**:
- Can we overlap CPU and GPU work using MLX streams?
- What's the optimal chunk size for sequence processing?
- How does precomputation affect memory vs recomputation tradeoff?

**Key Insights**:
- MLX lazy evaluation naturally overlaps operations
- Explicit streams beneficial for long sequences (>8K tokens)
- Chunk size 64-128 balances memory and recomputation

**Status**: Insights integrated into production MAD wiring scheduler

### 5. PyTorch MPS Kernel Bridge

**Location**: `pytorch_metal_kernels_demo.py`, `torch_msltm_kernels/`

**Purpose**: Demonstrate how to bridge custom Metal kernels into PyTorch MPS backend

**Approach**: 
- Embed Metal code as string literals in `.mm` files
- Runtime compilation via `newLibraryWithSource`
- PyTorch custom op registration

**Educational Value**: Shows MPS extension patterns but not used in production (MLX-first architecture)

**Status**: Archived reference for PyTorch developers

### 6. Orthogonality and SVD Kernels

**Location**: `mlx_orthogonality.py`, `naive_svd_at_a_v.py`

**Research**: Fast orthogonalization for weight matrices using Metal-accelerated SVD

**Status**: Experimental - not required for inference but useful for training/fine-tuning research

## Using Experimental Kernels

### Running Benchmarks

```bash
# Multi-head LayerNorm benchmark
cd experimental_kernels/2025-09-10-mlx-research
python mhln_bench.py

# GEMM tiling benchmark
python gemm_tile_bench.py



### Running Tests

```bash
# GEMM kernel validation
python experimental_kernels/test_gemm_kernels.py

# MLX research parity tests
python experimental_kernels/2025-09-10-mlx-research/run_test_mlx_parity.py
```

### Integration Guidelines

If you want to integrate an experimental kernel:

1. **Verify Numerical Parity**: Run tests and confirm error < 1e-6
2. **Benchmark**: Compare against production implementation on target hardware
3. **Feature Flag**: Add behind runtime flag for safe rollout
4. **Documentation**: Update technical docs with kernel description
5. **Fallback**: Ensure graceful fallback to MLX ops if kernel fails

## Performance Notes

**Apple Silicon Architecture**:
- Unified memory eliminates CPU-GPU transfer overhead
- Simdgroups (32 threads) are the fundamental execution unit
- Threadgroup memory (32KB) for fast inter-thread communication
- MLX automatically optimizes for M1/M2/M3/M4 generations

**Optimization Priorities**:
1. Memory access patterns (coalescing, bank conflicts)
2. Reduction locality (minimize threadgroup barriers)
3. Register pressure (limit per-thread memory)
4. Arithmetic intensity (compute vs memory ratio)

See [kernel_guide.md](docs/kernel_guide.md) for detailed optimization techniques.

## Documentation

### Comprehensive Guides

- **[kernel_guide.md](docs/kernel_guide.md)**: Complete Metal kernel development guide
  - MLX Metal kernels with `@mx.custom_function`
  - PyTorch MPS backend patterns
  - Metal Shading Language basics
  - Synchronization and memory management

- **[MLX_IMPLEMENTATION_GUIDE.md](docs/MLX_IMPLEMENTATION_GUIDE.md)**: MLX-specific patterns
  - Lazy evaluation and graph optimization
  - Custom function decorators
  - Metal kernel integration
  - Numerical stability techniques

### Research Reports

- **[docs/MLX_MHLN_Kernel_Report.md](docs/MLX_MHLN_Kernel_Report.md)**: Detailed analysis of multi-head LayerNorm optimization
- **[docs/2025-09-10-mlx-research-journal.md](docs/2025-09-10-mlx-research-journal.md)**: Daily research notes and findings
- **[GEMM_TILING_ATTEMPTS.md](matrix/gemm/GEMM_TILING_ATTEMPTS.md)**: GEMM optimization journey

## Contributing Experimental Kernels

When adding new experimental kernels:

1. **Create dated directory**: `YYYY-MM-DD-description/`
2. **Include README**: Describe approach, benchmarks, and findings
3. **Add tests**: Numerical parity and performance tests
4. **Document learnings**: What worked, what didn't, and why
5. **Update this index**: Add entry to directory structure and key research areas

## Production Path

For production xLSTM inference, use:

- **Core Implementation**: `xlstm_metal/blocks/mlstm_mlx/`
- **MAD Wiring**: `xlstm_metal/wiring/mlx/`
- **Entry Point**: `generate.py` or `from xlstm_metal import xLSTMRunner`

See main [README.md](../README.md) and [docs/](../docs/) for production documentation.

---

*These experimental kernels represent the research and optimization process behind xLSTM-Metal. They are preserved for education, benchmarking, and future research directions.*
