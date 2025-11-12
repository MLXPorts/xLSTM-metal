# M2-BERT Architecture Analysis

## Executive Summary

M2-BERT (Monarch Mixer BERT) is another architecture that uses **sequential block execution** with **within-layer
parallelism**. Like MAD, LFM2, and xLSTM, it does NOT have block-to-block parallelism.

**Key Innovation**: Uses **Monarch matrices** (block-diagonal structure) and **Hyena filters** (FFT-based long
convolutions) for sub-quadratic complexity.

**Parallelism Type**: GPU-level within layers (block-diagonal multiply, FFT channel batching)

---

## M2-BERT Architecture

### Model Structure

From `m2bert_model_mlx.py`:

```python
class M2BERTModel(nn.Module):
    def __init__(self, num_hidden_layers=12, ...):
        # Embeddings
        self.embeddings = BertEmbeddings(...)

        # Sequential layers (NOT parallel!)
        self.layers = []
        for i in range(num_hidden_layers):
            layer = M2BERTLayer(...)
            self.layers.append(layer)

    def __call__(self, input_ids):
        # Embeddings
        hidden_states = self.embeddings(input_ids)

        # Pass through all layers SEQUENTIALLY
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states
```

**Result**: Sequential execution, same as MAD/LFM2/xLSTM!

### Layer Structure

```python
class M2BERTLayer(nn.Module):
    """
    One M2-BERT layer: Monarch Mixer + Monarch MLP

    Architecture:
        Input
          ↓
        Monarch Mixer (sequence mixing, replaces attention)
          ↓
        Residual + LayerNorm
          ↓
        Monarch MLP (state mixing)
          ↓
        Output
    """

    def __call__(self, hidden_states):
        # Sequence mixing (Monarch Mixer with Hyena filters)
        seq_output, _ = self.sequence_mixing(hidden_states)
        hidden_states = self.layernorm_seq(hidden_states + seq_output)

        # State mixing (Monarch MLP)
        output = self.mlp(hidden_states)

        return output
```

**Pattern**: Pre-norm + operator + residual (same as LFM2!)

---

## Model Parameters

From `M2BERT_FULL_MODEL_COMPLETE.md`:

**M2-BERT-80M**:

- **Total parameters**: 81.3M (26% fewer than BERT-base)
- **Layers**: 12
- **Hidden size**: 768
- **Max sequence length**: 2048 (vs BERT's 512)
- **Monarch blocks**: 4 (for block-diagonal matrices)

**Parameter Breakdown**:

| Component       | Parameters     | % of Total |
|-----------------|----------------|------------|
| Embeddings      | 23,841,024     | 29.3%      |
| One M2BERTLayer | 4,690,944      | 5.8%       |
| 12 Layers       | 56,291,328     | 69.2%      |
| LayerNorm       | 15,872         | <0.1%      |
| **Total**       | **81,302,016** | **100%**   |

---

## Key Innovations

### 1. Monarch Matrices (Block-Diagonal Structure)

**Purpose**: Reduce parameters by 4x while maintaining expressivity

**Implementation**: `blockdiag_multiply_mlx.py`

```python
def blockdiag_multiply_reference(x, weight):
    """
    Block-diagonal matrix multiply

    Args:
        x: (batch, seq, d_model)  # d_model = 768
        weight: (nblocks, block_size, block_size)  # (4, 192, 192)

    Returns:
        out: (batch, seq, d_model)
    """
    batch, seqlen, d_model = x.shape
    nblocks, block_size, _ = weight.shape

    # Reshape into blocks
    x_blocks = x.reshape(batch, seqlen, nblocks, block_size)  # (B, S, 4, 192)

    # Apply each block matrix independently (PARALLEL on GPU!)
    out_blocks = []
    for i in range(nblocks):
        # Each block is independent - GPU can parallelize this
        block_out = mx.matmul(x_blocks[:, :, i, :], weight[i])  # (B, S, 192) @ (192, 192)
        out_blocks.append(block_out)

    # Concatenate blocks
    out = mx.concatenate(out_blocks, axis=-1)  # (B, S, 768)
    return out
```

**Parallelism**: Each block computed independently → GPU batched operations

**Parameter Reduction**:

- Standard: 768 × 768 = 589,824 params
- Monarch (4 blocks): 4 × 192 × 192 = 147,456 params (4x reduction)

### 2. Hyena Filters (FFT Long Convolution)

**Purpose**: O(L log L) sequence mixing (vs O(L²) attention)

**Implementation**: `monarch_mixer_mlx.py` + `hyena_filter_mlx.py`

Architecture:

```
Input
  ↓
Input Projection (3 gates: u, v, k)
  ↓
Short Convolution (local processing)
  ↓
Long Convolution via FFT (global context)
  ↓
Element-wise gating
  ↓
Output Projection
```

**FFT Convolution**:

```python
def fftconv_ref(u, k, D, dropout=0.0, gelu=True):
    """
    FFT-based convolution for long-range dependencies

    Args:
        u: (batch, d_model, seqlen) - input
        k: (d_model, seqlen) - learned filter
        D: (d_model,) - bias

    Returns:
        y: (batch, d_model, seqlen) - convolved output
    """
    seqlen = u.shape[2]
    fft_size = 2 * seqlen

    # Channel batching to avoid Metal's 499MB limit
    CHANNEL_BATCH_SIZE = 64
    outputs = []

    for ch_start in range(0, d_model, CHANNEL_BATCH_SIZE):
        ch_end = min(ch_start + CHANNEL_BATCH_SIZE, d_model)

        # FFT on input and kernel
        u_f = mx.fft.rfft(u[:, ch_start:ch_end, :], n=fft_size, axis=-1)
        k_f = mx.fft.rfft(k[ch_start:ch_end, :], n=fft_size, axis=-1)

        # Pointwise multiply (convolution in frequency domain)
        y_f = u_f * k_f[None, :, :]

        # Inverse FFT
        y = mx.fft.irfft(y_f, n=fft_size, axis=-1)[..., :seqlen]

        outputs.append(y)

    # Concatenate channel batches
    y = mx.concatenate(outputs, axis=1)

    # Add bias
    y = y + D[None, :, None]

    return y
```

**Parallelism**:

1. FFT is inherently parallel (O(L log L))
2. Channel batching allows parallel processing of channel subsets
3. All on GPU

---

## Metal FFT Solution (Critical Innovation)

From `METAL_FFT_SOLUTION.md`:

### The Problem

Metal has a **499MB per-allocation limit**. Large FFT convolutions exceeded this:

```python
# Configuration:
batch_size = 16
seq_len = 512
d_model = 768
num_layers = 12

# FFT creates: (16, 768, 1024) = ~50MB per layer × 12 layers
# With gradients: ~600MB total → EXCEEDS LIMIT
```

### The Solution: Channel Batching

**Key Insight**: Don't chunk the sequence (breaks algorithm) - batch orthogonal dimensions!

**Rule from HPC16x8**:
| Operation | Can Chunk Sequence? | Can Batch Channels? |
|-----------|-------------------|-------------------|
| FFT Convolution | ❌ No (global operation) | ✅ Yes (independent) |
| Matrix Multiply | ✅ Yes (with overlap-add) | ✅ Yes |
| Attention | ❌ No (quadratic dependencies) | ✅ Yes (heads) |

**Implementation**:

```python
CHANNEL_BATCH_SIZE = 64  # Process 64 channels at a time

for ch_start in range(0, d_model, CHANNEL_BATCH_SIZE):
    # Process channel batch (full sequence each time!)
    u_batch = u[:, ch_start:ch_end, :]  # (batch, 64, seqlen)
    k_f_batch = k_f[ch_start:ch_end, :]  # (64, fft_bins)

    # Full FFT convolution within batch
    u_f = mx.fft.rfft(u_batch, n=fft_size, axis=-1)
    y_f = u_f * k_f_batch[None, :, :]
    y_batch = mx.fft.irfft(y_f, n=fft_size, axis=-1)

    outputs.append(y_batch)

# Concatenate results (mathematically identical!)
y = mx.concatenate(outputs, axis=1)
```

**Result**:

- ✅ Perfect numerical correctness (max error = 0.0e+00)
- ✅ Memory efficient (~50MB per layer vs ~500MB)
- ✅ Minimal overhead (~5-10% latency from batching loop)

### HPC16x8 Patterns for Metal

From ember-ml's high-precision computing implementation:

1. **Explicit memory management**: Use `threadgroup` for shared memory
2. **Tiling over chunking**: Process tiles, not independent chunks
3. **SIMD operations**: Leverage warp-level primitives
4. **Thread barriers**: Synchronize with `threadgroup_barrier`
5. **Limb-based arithmetic**: For extended precision (8 limbs)

Example Metal kernel structure:

```metal
#define NUM_LIMBS 8
#define TILE_SIZE 256
#define WARP_SIZE 32

threadgroup float shared_Z[250 * MAX_K];  // Explicit memory management
threadgroup float shared_proj[MAX_K];
threadgroup float shared_norm[MAX_K];

kernel void fft_conv_tiled(
    device const float* u,
    device const float* k_f_real,
    device const float* k_f_imag,
    device float* y,
    threadgroup float2* shared_u_f,
    uint tid [[thread_position_in_grid]]
) {
    // Process in tiles
    for (uint tile = 0; tile < num_tiles; tile++) {
        // Load tile into shared memory
        // Process with SIMD operations
        // Synchronize with barriers
    }
}
```

---

## Integration with xLSTM Kernels

**M2-BERT already uses xLSTM Metal GEMM kernels!**

From `blockdiag_multiply_mlx.py`:

```python
# Lines 14-18: Import xLSTM kernels
xlstm_kernels_path = '<workspace_root>/xLSTM/experimental_kernels/mlx_fast_kernels'
if xlstm_kernels_path not in sys.path:
    sys.path.insert(0, xlstm_kernels_path)
from gemm_kernels import gemm_av, gemm_at_b

# Lines 57-107: Fast GEMM implementation
def blockdiag_multiply_gemm(x, weight):
    """
    Block-diagonal multiply using Metal-accelerated GEMM from xLSTM

    Faster than reference for large matrices (2-3x speedup)
    """
    batch, seqlen, d_model = x.shape
    nblocks, block_size, _ = weight.shape

    # Reshape into blocks
    x_blocks = x.reshape(batch, seqlen, nblocks, block_size)

    result_blocks = []
    for i in range(nblocks):
        # Use xLSTM's optimized Metal GEMM kernel
        x_block = x_blocks[:, :, i, :]  # (B, S, block_size)
        W_block = weight[i]  # (block_size, block_size)

        # Reshape for gemm_av: expects (batch, block_size, seqlen)
        x_block_T = x_block.transpose(0, 2, 1)

        # Metal-accelerated matrix multiply
        result = gemm_av(W_block, x_block_T)  # (batch, block_size, seqlen)

        # Transpose back
        result = result.transpose(0, 2, 1)  # (batch, seqlen, block_size)
        result_blocks.append(result)

    # Concatenate blocks
    out = mx.concatenate(result_blocks, axis=-1)
    return out
```

**Performance**: 2-3x faster than reference for large matrices

---

## Parallelism Summary

### What M2-BERT Has (Within-Layer)

1. **Block-Diagonal Parallelism**
    - Each Monarch block computed independently
    - GPU batched operations across blocks
    - 4x parameter reduction with maintained capacity

2. **Channel Batching (FFT Convolution)**
    - Process 64 channels at a time
    - Full FFT convolution per batch
    - Memory-efficient (avoids Metal's 499MB limit)

3. **FFT Parallelism**
    - Inherently parallel algorithm (O(L log L))
    - GPU-optimized FFT primitives
    - Sub-quadratic complexity vs O(L²) attention

4. **xLSTM GEMM Kernels**
    - Metal-accelerated matrix multiply
    - 2-3x faster than reference
    - Shared with xLSTM project

### What M2-BERT Does NOT Have

❌ **Block-to-Block Parallelism** - Layers execute sequentially:

```python
# Sequential iteration (same as MAD/LFM2/xLSTM!)
for layer in self.layers:
    hidden_states = layer(hidden_states)
```

❌ **Threading/Async** - No multi-threaded block execution

❌ **DAG-based Architecture** - Linear dependency chain

---

## Implications for xLSTM-MAD-NCPS

### What We Can Adopt

1. **Hyena Filters as Block Type**
   ```python
   # Mixed architecture
   layer_types = [
       "mlstm",      # xLSTM layer
       "hyena",      # Hyena FFT convolution (from M2-BERT)
       "slstm",      # sLSTM variant
       "mamba"       # State-space model
   ]
   ```

2. **Metal FFT Convolution**
    - Channel batching pattern for memory efficiency
    - HPC16x8 threadgroup memory management
    - Custom Metal kernels for 2-5x speedup

3. **Block-Diagonal Matrices**
    - Potential for parameter-efficient variants
    - Could reduce mLSTM projection sizes
    - Monarch pattern for Q/K/V projections

4. **xLSTM GEMM Kernel Reuse**
    - M2-BERT already uses our kernels!
    - Proven 2-3x speedup
    - Could integrate bidirectionally

### Implementation Strategy

**Phase 1: Core Architecture** (matches previous analysis)

1. Fix dtype issue in mLSTM kernel
2. Create BlockRegistry (mLSTM, sLSTM, FFN types)
3. Implement config-driven backbone (LFM2 `layer_types` pattern)

**Phase 2: M2-BERT Integration**

4. Add Hyena block type
    - Use M2-BERT's `MonarchMixerSequenceMixing`
    - Integrate FFT channel batching
    - Add to BlockRegistry

5. Create Metal FFT utilities
    - Standalone `metal_fft_conv.py` (like M2-BERT)
    - HPC16x8 threadgroup patterns
    - Custom kernels for 2-5x speedup

6. Optional: Monarch matrix projections
    - Block-diagonal Q/K/V for mLSTM
    - 4x parameter reduction
    - Experimental variant

**Phase 3: Cross-Pollination**

7. Share xLSTM GEMM kernels with M2-BERT team
8. Integrate M2-BERT's Hyena improvements back to xLSTM
9. Unified Metal kernel library

---

## Conclusion

M2-BERT confirms the universal pattern:

**ALL modern architectures use:**

- ✅ Sequential block execution
- ✅ Within-layer parallelism (GPU-level)
- ✅ Residual connections with pre-normalization

**Parallelism hierarchy:**

1. **GPU-level**: Threadgroups, SIMD, tiling (Metal kernels)
2. **Algorithm-level**: FFT (O(L log L)), block-diagonal, multi-head
3. **Batching-level**: Channel batching, expert routing
4. **NOT block-to-block**: Sequential dependencies prevent this

**Key takeaway**: Don't try to parallelize block execution - focus on optimizing within-layer computation through:

- Efficient kernels (Metal, custom GEMM)
- Smart memory management (channel batching, tiling)
- Algorithmic efficiency (FFT vs direct convolution, sub-quadratic attention)

**For xLSTM-MAD-NCPS**, the path forward is:

1. Keep simple sequential backbone (like MAD/LFM2/M2-BERT)
2. Add heterogeneous block types (Hyena, Mamba, NCPS mixers)
3. Optimize within-layer computation (Metal kernels, FFT)
4. Use NCPS wiring INSIDE blocks for component-level flexibility

---

## Files Referenced

**M2-BERT Implementation**:

- `m2_training/src/mm_mlx/m2bert_model_mlx.py` - Main model
- `m2_training/src/mm_mlx/monarch_mixer_mlx.py` - Hyena sequence mixing
- `m2_training/src/mm_mlx/hyena_filter_mlx.py` - FFT convolution
- `m2_training/src/mm_mlx/blockdiag_multiply_mlx.py` - Block-diagonal multiply (uses xLSTM kernels!)

**Documentation**:

- `m2_training/M2BERT_FULL_MODEL_COMPLETE.md` - Architecture overview
- `m2_training/METAL_FFT_SOLUTION.md` - Channel batching solution

**xLSTM Kernel Integration**:

- `<workspace_root>/xLSTM/experimental_kernels/mlx_fast_kernels/gemm_kernels.py`

**HPC16x8 Reference**:

- `<workspace_root>/Projects/ember-ml-kotlin/ember_ml/backend/mlx/linearalg/hpc16x8_ops.py`
- `<workspace_root>/Projects/ember-ml-kotlin/ember_ml/backend/mlx/linearalg/svd_ops.py`
