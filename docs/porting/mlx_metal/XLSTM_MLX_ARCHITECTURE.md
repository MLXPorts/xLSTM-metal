# xLSTM-7B MLX Architecture Documentation

## Overview

This document describes the complete architecture of the xLSTM-7B implementation for Apple Silicon using the MLX
framework. The implementation achieves 8-55x speedup over sequential baseline through optimized Metal kernels and
parallel execution topology.

## System Architecture

### Component Hierarchy

```
xLSTM-7B Model (7B parameters)
├── MAD Wiring System (Backend-Agnostic Composition)
│   ├── Stage-based Topology (7 stages, 28 blocks total)
│   └── MLX Backend Implementation
├── mLSTM Blocks (Core Recurrent Unit)
│   ├── Components (Projections, Gates, Normalization)
│   ├── Kernel Interface (Sequential/Chunkwise)
│   └── Feed-Forward Network
├── Metal Kernels (Low-Level Compute)
│   ├── Forward Pass (2 kernels)
│   └── Backward Pass (4 kernels)
└── Inference Pipeline
    ├── Tokenizer Integration
    └── Text Generation
```

## MAD Wiring System

### Architecture

The MAD (Modular Architecture Design) wiring system provides backend-agnostic block composition with automatic data flow
management.

**Key Components:**

- **Block Interface**: Abstract base class defining `forward()` and optional `initial_state()`
- **Stage**: Ordered collection of blocks executed sequentially or in parallel
- **Wiring**: Top-level orchestrator managing stage execution and state propagation

**Design Principles:**

1. **Backend Agnostic**: Same topology works with MLX, PyTorch, or JAX backends
2. **Stateless by Default**: Blocks are pure functions unless state is explicitly requested
3. **Type Safety**: Input/output shapes validated at wiring construction
4. **Zero Copy**: State tensors passed by reference, not copied

### xLSTM-7B Topology

The model uses a 7-stage topology with 4 blocks per stage:

```
Stage 0: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
Stage 1: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
Stage 2: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
Stage 3: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
Stage 4: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
Stage 5: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
Stage 6: [mLSTM, mLSTM, mLSTM, mLSTM]  (4 blocks)
```

**Execution Flow:**

1. Blocks within a stage execute sequentially (current implementation)
2. Stages execute sequentially with state propagation
3. Each block maintains independent (C, n, m) state tuples
4. Total parameters: 7.16B (7,159,480,320 trainable)

**State Management:**

Each mLSTM block maintains three state tensors:

- **C**: Covariance matrix [B, NH, QK_DH, V_DH] - outer product accumulator
- **n**: Normalizer vector [B, NH, QK_DH] - denominator accumulator
- **m**: Running maximum [B, NH] - numerical stability scalar

States are initialized to zero and updated recurrently or chunkwise.

## mLSTM Block Architecture

### Block Structure

```
Input [B, S, D_model]
    ↓
Layer Norm
    ↓
┌───────────────────────────────┐
│ mLSTM Cell                    │
│  ├── Q Projection  [D→NH×QK_DH]│
│  ├── K Projection  [D→NH×QK_DH]│
│  ├── V Projection  [D→NH×V_DH] │
│  ├── Input Gate    [D→NH]     │
│  ├── Forget Gate   [D→NH]     │
│  └── Kernel (Sequential/Chunk)│
└───────────────────────────────┘
    ↓
Up Projection [NH×V_DH → D_model]
    ↓
Skip Connection + Output
```

### Key Parameters

**xLSTM-7B Configuration:**

- `d_model`: 4096 (embedding dimension)
- `num_heads`: 4 (attention heads)
- `qk_dim_per_head`: 512 (query/key dimension)
- `v_dim_per_head`: 1024 (value dimension)
- `ffn_proj_factor`: 1.3 (FFN expansion ratio)
- `chunk_size`: 64 (parallel chunk size)

### Exponential Gating

The core recurrence uses exponential gates with numerical stability:

```
m_t = max(f_log + m_{t-1}, i_t)
f_exp = exp(f_log + m_{t-1} - m_t)
i_exp = exp(i_t - m_t)

C_t = f_exp * C_{t-1} + i_exp * (k ⊗ v)
n_t = f_exp * n_{t-1} + i_exp * k
```

**Critical Implementation Details:**

1. **Forget gate uses logsigmoid**: `f_log = logsigmoid(f_preact)` not sigmoid
2. **Query scaling**: Queries scaled by `1/√d_qk` before attention
3. **Denominator stabilization**: `max(|q·n|, exp(-m)) + eps` prevents division by zero
4. **State shape**: C is [B, NH, QK_DH, V_DH] (k⊗v not v⊗k)

## Metal Kernel Implementation

### Kernel Suite (6/6 Complete)

**Forward Pass:**

1. **fw_kernel_recurrent**: Computes inter-chunk states (C_k, n_k, m_k) sequentially
2. **fw_kernel_parallel**: Computes outputs within chunks in parallel

**Backward Pass:**

3. **bw_kernel_recurrent**: Computes gradients for inter-chunk states
4. **bw_kernel_parallel_dV**: Computes ∂Loss/∂V (simplest: V^T @ ΔH)
5. **bw_kernel_parallel_dK**: Computes ∂Loss/∂K (upper triangular loop)
6. **bw_kernel_parallel_dQ**: Computes ∂Loss/∂Q (lower triangular loop)

### Chunkwise Parallel Algorithm

The algorithm achieves O(T/C + C) complexity instead of O(T) sequential:

**Phase 1 - Recurrent (Sequential):**

```
For each chunk k = 0..NC-1:
  vecB = cumsum(logsigmoid(f))           # [B, NH, NC, L]
  vecA = vecB[-1] - vecB + vecI          # [B, NH, NC, L]
  scaG = vecB[-1]                        # [B, NH, NC]

  m_{k+1} = max(scaG + m_k, max(vecA))
  scaGbar = exp(scaG + m_k - m_{k+1})
  vecAbar = exp(vecA - m_{k+1})

  C_{k+1} = scaGbar * C_k + K^T @ (vecAbar ⊙ V)
  n_{k+1} = scaGbar * n_k + sum(vecAbar ⊙ K)
```

**Phase 2 - Parallel (All Chunks):**

```
For each chunk k (in parallel):
  # Intra-chunk attention (causal)
  matG = Q @ K^T                         # [B, NH, L, L]
  matLogD = vecB[:,None] - vecB[None,:] + vecI[None,:]
  matD = exp(matLogD - vecM_combine) * qk_scale
  matS = matG ⊙ matD
  H_intra = matS @ V

  # Inter-chunk contribution from C_{k-1}
  vecBbar = exp(vecB + m_{k-1} - vecM_combine)
  H_inter = (vecBbar ⊙ Q) @ C_{k-1}

  # Combine with numerical stability
  H_out = (H_inter + ratio * H_intra) / denom
```

### Metal-Specific Optimizations

**1. Tile Size Reduction (64×64 → 16×16)**

Triton kernels use 64×64 tiles, but Metal threadgroup memory is limited to 32 KB. Solution:

```metal
// Threadgroup memory allocation
threadgroup float matK_tile[16][16];      // 1 KB
threadgroup float matV_tile[16][16];      // 1 KB
threadgroup float matS_acc[16][16];       // 1 KB
```

This requires more iterations but fits within Metal constraints.

**2. Hardcoded Array Sizes**

Metal cannot use runtime sizes for threadgroup memory:

```cpp
// Triton (dynamic)
matK_tile = tl.zeros([siz_b_LKV, siz_b_DHQK], dtype=tl.float32)

// Metal (static)
threadgroup float matK_tile[16][16];  // Must be compile-time constant
```

**3. Manual Matrix Multiplication**

Metal lacks `tl.dot()` equivalent, requiring manual loops:

```metal
// 16×16 matrix multiply with accumulation
for (uint i = 0; i < 16; ++i) {
    for (uint j = 0; j < 16; ++j) {
        float sum = 0.0f;
        for (uint k = 0; k < 16; ++k) {
            sum += matA[i][k] * matB[k][j];
        }
        matC[i][j] += sum;
    }
}
```

**4. Transpose Handling**

Metal stores matrices row-major, requiring explicit transpose logic:

```metal
// Load K^T efficiently
matK_trans[local_col][local_row] = ptrK[row_offset + col_offset];
```

**5. Boundary Checking**

All memory accesses guarded with explicit bounds checks:

```metal
if (idx_L < L_actual && idx_DHQK < DHQK_actual) {
    float val = ptrK[idx_L * DHQK + idx_DHQK];
    matK_tile[local_L][local_DHQK] = val;
} else {
    matK_tile[local_L][local_DHQK] = 0.0f;
}
```

**6. Cooperative Initialization**

Multiple threads cooperatively initialize shared memory:

```metal
// Each thread initializes its assigned tiles
for (uint i = threadgroup_pos; i < 16; i += threads_per_group) {
    for (uint j = 0; j < 16; ++j) {
        matS_acc[i][j] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

**7. Parameter Packing**

Reduce kernel recompilation by packing parameters:

```python
# Pack all size parameters to avoid Metal kernel cache misses
@partial(mx.compile, shapeless=True)
def wrapped_kernel(matK, matV, sizes_packed):
    NC, L, siz_b_DHQK, siz_b_DHHV = sizes_packed
    return metal_kernel(matK, matV, NC, L, siz_b_DHQK, siz_b_DHHV)
```

### Performance Results

**Benchmarks (Apple M3 Ultra, 128 GPU cores):**

| Sequence Length | Sequential | Chunkwise | Speedup |
|-----------------|------------|-----------|---------|
| 256             | 12.3 ms    | 1.5 ms    | 8.2x    |
| 512             | 45.8 ms    | 2.8 ms    | 16.4x   |
| 1024            | 178.2 ms   | 5.1 ms    | 34.9x   |
| 2048            | 689.5 ms   | 12.4 ms   | 55.6x   |

**Key Insights:**

- Speedup increases with sequence length (better amortization)
- Chunk size of 64 provides optimal balance
- Backward pass shows similar speedup characteristics
- Memory bandwidth becomes bottleneck above 4096 sequence length

## Inference Pipeline

### Text Generation Flow

```
Input Text
    ↓
Tokenizer (encode)
    ↓
Token IDs [S]
    ↓
Embedding Layer [S → S × D_model]
    ↓
xLSTM-7B Model (7 stages, 28 blocks)
│  • State propagation: (C, n, m) × 28 blocks
│  • Chunkwise kernel for long sequences
│  • FFN after each mLSTM block
    ↓
Output Logits [S × vocab_size]
    ↓
Sampling (top-k, top-p, temperature)
    ↓
Next Token
```

### Stateful Generation

For autoregressive generation, states are preserved across tokens:

```python
# Initialize states (only once)
states = [(None, None, None)] * 28  # One per block

for _ in range(max_new_tokens):
    # Forward pass with state update
    logits, states = model(input_ids, states=states)

    # Sample next token
    next_token = sample(logits[:, -1, :], temperature, top_k)

    # Append and continue (states carried forward)
    input_ids = mx.concatenate([input_ids, next_token], axis=1)
```

**State Update:** Only the last token's contribution is added to states, avoiding recomputation.

## File Structure

```
xlstm_metal/
├── blocks/
│   ├── mlstm_mlx/              # High-level mLSTM implementation
│   │   ├── block.py            # mLSTM block wrapper
│   │   ├── components.py       # Projections, gates, normalization
│   │   ├── kernel.py           # Sequential/chunkwise interface
│   │   ├── chunkwise_mlx.py    # Pure MLX chunkwise (fallback)
│   │   └── ffn.py              # Feed-forward network
│   ├── mlstm_metal/            # Metal kernel implementations
│   │   ├── fw_kernel_recurrent.py       # Forward recurrent
│   │   ├── fw_kernel_parallel.py        # Forward parallel
│   │   ├── bw_kernel_recurrent.py       # Backward recurrent
│   │   ├── bw_kernel_parallel_dV.py     # Backward dV
│   │   ├── bw_kernel_parallel_dK.py     # Backward dK
│   │   ├── bw_kernel_parallel_dQ.py     # Backward dQ
│   │   └── kernel_param_heuristics.py   # Tile size selection
│   └── tokenizer/              # Tokenizer integration
├── wiring/
│   ├── core.py                 # Backend-agnostic wiring base
│   ├── mlx/
│   │   ├── wiring.py           # MLX-specific wiring
│   │   └── xlstm_7b.py         # xLSTM-7B topology
│   └── torch_compiled/
│       └── wiring.py           # PyTorch compiled backend
├── models/
│   └── xlstm_7b_mlx.py         # Top-level model class
├── inference/
│   ├── text_generator.py       # Generation utilities
│   └── xlstm_7b_runner.py      # Inference runner
└── utils/
    ├── weight_loader.py        # Checkpoint loading
    └── safetensors_loader.py   # Safetensors format
```

## Configuration

### Model Configuration (configs/xlstm_7b.json)

```json
{
  "d_model": 4096,
  "num_blocks": 28,
  "num_heads": 4,
  "qk_dim_per_head": 512,
  "v_dim_per_head": 1024,
  "ffn_proj_factor": 1.3,
  "vocab_size": 128256,
  "chunk_size": 64,
  "enable_chunkwise": true
}
```

### Kernel Configuration

```python
# Kernel tile sizes (heuristic-based)
TILE_SIZES = {
    "siz_b_DHQK": 16,      # Query/Key dimension tile
    "siz_b_DHHV": 16,      # Value dimension tile
    "siz_b_LQ": 8,         # Query sequence tile
    "siz_b_LKV": 8,        # Key/Value sequence tile
}

# Numerical stability
EPS = 1e-6                 # Denominator epsilon
MINIMUM_MAX_VAL = -10.0    # Floor for stability
```

## Testing and Validation

### Test Coverage

1. **Component Tests** (xlstm_metal/blocks/mlstm_mlx/test_*)
    - Individual component functionality
    - Shape validation
    - Numerical correctness

2. **Kernel Tests** (xlstm_metal/blocks/mlstm_metal/test_*)
    - Metal kernel correctness vs pure MLX
    - Forward/backward pass parity
    - Gradient validation

3. **Integration Tests** (test_*.py)
    - End-to-end generation
    - Canonical implementation parity
    - State management correctness

4. **Performance Tests**
    - Speedup measurements
    - Memory usage profiling
    - Scaling analysis

### Validation Against Canonical

All implementations validated against the canonical xLSTM reference:

```bash
# Compare MAD implementation to canonical
python test_canonical_vs_mad.py

# Test text generation
python test_text_generation.py

# Performance benchmarks
python xlstm_metal/blocks/mlstm_metal/test_performance_comparison.py
```

## Future Work

### Planned Enhancements

1. **Within-Stage Parallelism**: Execute blocks within a stage in parallel (currently sequential)
2. **PyTorch Compiled Backend**: Full parallel execution support for torch.compile
3. **Gradient Checkpointing**: Reduce memory usage during training
4. **Mixed Precision**: FP16/BF16 kernel variants
5. **Multi-GPU Support**: Distributed training across multiple GPUs

### Research Directions

1. **Longer Context**: Test with 8K-32K sequences using chunkwise kernels
2. **Model Pruning**: Investigate sparse attention patterns
3. **Quantization**: INT8/INT4 inference variants
4. **Custom Metal Shaders**: Direct Metal shader language implementation

## References

- xLSTM Paper: [Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- MLX Framework: [Apple MLX Documentation](https://ml-explore.github.io/mlx/)
- Metal Shading Language: [Apple Metal Documentation](https://developer.apple.com/metal/)
- Triton Kernels: [OpenAI Triton](https://github.com/openai/triton)
