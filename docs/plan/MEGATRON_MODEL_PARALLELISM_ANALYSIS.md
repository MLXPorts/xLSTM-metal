# Megatron-LM Model Parallelism Analysis for xLSTM

**Intra-layer model parallelism for scaling beyond single-GPU memory limits**

Date: 2025-01-21
Source: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
Relevance: Scaling xLSTM training beyond Apple Silicon single-GPU limits

## Executive Summary

Megatron-LM demonstrates **intra-layer model parallelism** for transformers, achieving:
- **8.3B parameters** on 512 GPUs (8-way model parallel × 64-way data parallel)
- **76% scaling efficiency** vs single GPU baseline
- **Simple implementation** - no compiler changes, just a few communication ops
- **Orthogonal to pipeline parallelism** (can combine both)

**Key insight:** Split attention heads and MLP GEMMs **within layers** rather than layers across GPUs.

**Comparison to Hogwild!:**
- **Hogwild!**: Lock-free data parallelism (multiple copies of model)
- **Megatron**: Model parallelism (split single model across GPUs)
- **Combined approach**: Both techniques work together for massive scale

## Core Technique: Intra-Layer Model Parallelism

### Problem
Large transformers exceed single GPU memory:
- Model weights
- Optimizer state (Adam: 2× per parameter)
- Activations
- Gradients

**Example:** 8.3B parameter model requires ~33GB just for weights (fp32) or ~16.5GB (fp16)
- Add Adam state: +33GB
- Add activations: +10-20GB
- **Total: 60-70GB** >> 32GB V100 memory

### Solution: Split Layers Horizontally

Instead of putting different layers on different GPUs (pipeline parallelism), **split each layer across GPUs**.

## Transformer Model Parallelism Details

### 1. MLP Block Parallelization

**MLP structure:**
```
Y = Dropout(GELU(X @ A) @ B)
```

**Column-parallel first GEMM:**
```python
# Split A along columns: A = [A1, A2]
# Each GPU computes independently
Y1 = GELU(X @ A1)  # GPU 1
Y2 = GELU(X @ A2)  # GPU 2
# Y = [Y1, Y2] (concatenated)
```

**Row-parallel second GEMM:**
```python
# Split B along rows: B = [B1; B2]
# Each GPU computes partial result
Z1 = Y1 @ B1  # GPU 1
Z2 = Y2 @ B2  # GPU 2
# All-reduce to get final: Z = Z1 + Z2
```

**Key benefit:** Only **one all-reduce** in forward pass (g operator), one in backward (f operator).

### 2. Self-Attention Block Parallelization

**Multi-head attention is naturally parallel!**

```python
# For 8 heads on 4 GPUs: 2 heads per GPU
# GPU 1: heads 0, 1
# GPU 2: heads 2, 3
# GPU 3: heads 4, 5
# GPU 4: heads 6, 7

# Each GPU computes Q, K, V for its heads independently
Q_local = X @ W_Q_local  # Column-parallel
K_local = X @ W_K_local
V_local = X @ W_V_local

# Attention computed independently per GPU
attention_local = softmax(Q_local @ K_local.T) @ V_local

# Output projection (row-parallel)
output_partial = attention_local @ W_O_local
# All-reduce across GPUs
output = all_reduce(output_partial)
```

**Key benefit:** Attention heads are **embarrassingly parallel** - no communication until final projection.

### 3. Communication Primitives

**f operator:**
```python
class f(torch.autograd.Function):
    def forward(ctx, x):
        return x  # Identity in forward

    def backward(ctx, grad):
        all_reduce(grad)  # Sum gradients
        return grad
```

**g operator:**
```python
class g(torch.autograd.Function):
    def forward(ctx, x):
        all_reduce(x)  # Sum activations
        return x

    def backward(ctx, grad):
        return grad  # Identity in backward
```

**Total communication per layer:**
- Forward: 2 all-reduce operations (MLP output + attention output)
- Backward: 2 all-reduce operations (gradients)
- **4 all-reduce ops per transformer layer**

### 4. Embedding Layer Parallelization

**Challenge:** Output embedding shares weights with input embedding, and vocabulary is large (50k tokens).

**Solution:**
```python
# Split vocabulary across GPUs
# GPU 1: tokens 0-12,799
# GPU 2: tokens 12,800-25,599
# GPU 3: tokens 25,600-38,399
# GPU 4: tokens 38,400-51,199

# Input embedding: all-reduce after lookup
embeddings_partial = lookup(token_ids, embedding_table_local)
embeddings = all_reduce(embeddings_partial)

# Output embedding: fuse with cross-entropy to reduce communication
logits_local = hidden @ vocab_weights_local.T
# Instead of all-gather(logits_local) → cross_entropy
# Do: cross_entropy_fused(logits_local, labels)
# Communicate scalar losses instead of |vocab|-dimensional logits!
```

**Key optimization:** Communicating **losses** (b × s scalars) instead of **logits** (b × s × v) reduces communication by factor of vocab_size.

## Scaling Results (from Paper)

### Model Parallel Scaling

| GPUs | Parameters | Speedup | Efficiency |
|------|------------|---------|------------|
| 1    | 1.2B       | 1.0×    | 100%       |
| 2    | 2.5B       | 1.95×   | 97%        |
| 4    | 4.2B       | 3.28×   | 82%        |
| 8    | 8.3B       | 6.16×   | 77%        |

**Single GPU baseline:** 39 TeraFLOPS (30% of V100 peak)

### Model + Data Parallel Scaling

| GPUs | Model Parallel | Data Parallel | Efficiency |
|------|----------------|---------------|------------|
| 64   | 1×             | 64×           | 96%        |
| 128  | 2×             | 64×           | 83%        |
| 256  | 4×             | 64×           | 79%        |
| 512  | 8×             | 64×           | 74%        |

**Key insight:** 76% efficiency at 512 GPUs is excellent - most of the loss is from data parallelism gradient sync, not model parallelism.

## Application to xLSTM on Apple Silicon

### Current Limits

**Apple M2 Ultra:**
- 24 cores (20 performance + 4 efficiency)
- 192GB unified memory
- 800 GB/s memory bandwidth

**Largest model that fits:**
- ~10B parameters (fp16) with activations/gradients
- With activation checkpointing: ~15B parameters
- Beyond this: Need model parallelism

### xLSTM Model Parallelism Strategy

**For xlstm-large (2048 dim, 48 blocks):**

**Memory breakdown (8.3B parameters, similar to Megatron's largest):**
```
Weights (fp16):          16.6 GB
Adam state (fp32):       33.2 GB
Activations (batch=32):  ~10 GB
Gradients:               16.6 GB
-----------------------------------
Total:                   ~76 GB
```

**Solution:** 4-way model parallelism
- Split across 4 GPUs (e.g., 4 Apple M2 Max in cluster)
- Each GPU: ~19 GB (fits in 32GB M2 Max)
- Or: Use 2 Apple M2 Ultra (2 GPUs each)

### Implementation for MLX

**MLX supports model parallelism via:**
1. **Unified memory architecture** - all GPUs share memory space
2. **Metal atomics** - for all-reduce operations
3. **Stream chaining** - overlap communication and computation

**Example implementation:**

```python
# mad/blocks/mlstm_mlx/model_parallel.py
import mlx.core as mx
from mlx.utils import tree_map

class ModelParallelmLSTMBlock(mx.nn.Module):
    def __init__(self, config, model_parallel_size=4):
        super().__init__()
        self.model_parallel_size = model_parallel_size
        self.rank = get_rank()  # Which GPU am I?

        # Split attention heads across GPUs
        self.num_heads_per_gpu = config.num_heads // model_parallel_size
        self.head_start = self.rank * self.num_heads_per_gpu
        self.head_end = (self.rank + 1) * self.num_heads_per_gpu

        # Column-parallel Q, K, V projections
        self.qkv_proj = mx.nn.Linear(
            config.d_model,
            3 * self.num_heads_per_gpu * config.head_dim,
            bias=False
        )

        # Row-parallel output projection
        self.out_proj = mx.nn.Linear(
            self.num_heads_per_gpu * config.head_dim,
            config.d_model,
            bias=False
        )

    def __call__(self, x, hidden_state):
        # QKV projection (column-parallel, no communication)
        qkv = self.qkv_proj(x)
        qkv = mx.reshape(qkv, (batch, seq_len, 3, self.num_heads_per_gpu, head_dim))
        q, k, v = mx.split(qkv, 3, axis=2)

        # Attention (local to this GPU)
        attn_output = self._compute_attention(q, k, v, hidden_state)

        # Output projection (row-parallel)
        output_partial = self.out_proj(attn_output)

        # All-reduce (g operator)
        output = all_reduce_mlx(output_partial)

        return output, new_hidden_state
```

**MLX all-reduce implementation:**
```python
def all_reduce_mlx(tensor, group_size=4):
    """All-reduce across model parallel GPUs using Metal atomics"""
    # Use MLX's distributed primitives (when available)
    # Or implement via shared memory + Metal atomics

    # For now, simulate with metal_kernel
    # In practice, MLX would provide this natively
    return mx.distributed.all_reduce(tensor, group=model_parallel_group)
```

### Combining Megatron + Hogwild!

**Powerful combination:**

1. **Model parallelism (Megatron):** Split model across GPUs
   - Enables training models larger than single GPU
   - 4-8 way parallelism typical

2. **Data parallelism (Hogwild!):** Multiple model replicas
   - Lock-free SGD updates
   - Near-linear speedup with sparse gradients

**Example configuration:**
```
Total GPUs: 32 (e.g., 4 Apple M2 Ultra machines × 2 GPUs × 4 replicas)

Model parallelism: 4-way (split model across 4 GPUs)
Data parallelism: 8-way (8 replicas of the 4-GPU model)

Effective speedup:
- Model parallel: 3-4× (allows larger model)
- Data parallel: 7-8× (Hogwild! speedup)
- Total: ~25-30× faster training
```

## Critical Implementation Details

### 1. Random Number Generation

**Problem:** Dropout must be consistent within model parallel group, random across data parallel groups.

**Solution:**
```python
# Global RNG (same seed across model parallel group)
global_rng = mx.random.seed(42)

# Local RNG (unique per GPU)
local_rng = mx.random.seed(42 + rank)

# Dropout outside model parallel regions: use global_rng
x = mx.random.dropout(x, rate=0.1, stream=global_rng)

# Dropout inside model parallel regions: use local_rng
x_parallel = mx.random.dropout(x_parallel, rate=0.1, stream=local_rng)
```

### 2. Activation Checkpointing

**Still needed with model parallelism!**

Even though weights are distributed, activations can still be large:
```python
# Checkpoint after each transformer layer
class CheckpointedTransformerLayer(mx.nn.Module):
    def __call__(self, x):
        # Don't store intermediate activations in forward
        # Recompute them in backward pass
        return mx.checkpoint(self._forward, x)
```

### 3. Vocabulary Padding

**For efficient GEMMs, pad vocabulary to be divisible by (128 × model_parallel_size):**

```python
# Original vocab: 50,257
# Model parallel: 8-way
# Required: divisible by 128 × 8 = 1024
# Padded vocab: 51,200

vocab_size_padded = math.ceil(vocab_size / (128 * model_parallel_size)) * 128 * model_parallel_size
```

This ensures each GPU gets a multiple of 128 tokens for efficient matrix multiplies.

### 4. Communication Optimization

**Key optimizations:**
1. **Overlapping communication and computation** (pipeline microbatching)
2. **Fusing operations** (combine multiple small all-reduces)
3. **Gradient bucketing** (reduce communication frequency)

```python
# Example: Overlap backward pass communication
async def backward_with_overlap():
    # Start backward pass
    grad = compute_gradient(layer_n)

    # Immediately start all-reduce (non-blocking)
    handle = all_reduce_async(grad)

    # Continue with next layer's backward pass
    grad_n_minus_1 = compute_gradient(layer_n_minus_1)

    # Wait for all-reduce to complete
    grad = wait_for_all_reduce(handle)

    # Apply gradient
    update_weights(grad)
```

## BERT Layer Normalization Fix

**Critical finding from paper:** Original BERT architecture degrades at large scales.

**Problem architecture:**
```python
# Original BERT (doesn't scale)
def bert_layer_original(x):
    # Post-LN (layer norm AFTER residual)
    attn_out = attention(x)
    x = layer_norm(x + attn_out)

    mlp_out = mlp(x)
    x = layer_norm(x + mlp_out)
    return x
```

**Fixed architecture:**
```python
# Megatron BERT (scales well)
def bert_layer_fixed(x):
    # Pre-LN (layer norm BEFORE attention/MLP)
    attn_out = attention(layer_norm(x))
    x = x + attn_out

    mlp_out = mlp(layer_norm(x))
    x = x + mlp_out
    return x
```

**Impact:** Enables scaling BERT from 336M to 3.9B parameters with monotonic improvement.

**Applies to xLSTM?** Yes! Our mLSTM blocks should use **pre-normalization**:

```python
# mad/blocks/mlstm_mlx/block.py
class mLSTMBlockMLX(mx.nn.Module):
    def __call__(self, x, hidden_state):
        # Pre-norm (like Megatron BERT)
        x_norm = self.input_norm(x)

        # mLSTM computation
        mlstm_out, hidden_state = self.mlstm(x_norm, hidden_state)

        # Residual connection
        x = x + mlstm_out

        # FFN with pre-norm
        x_norm = self.post_attn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, hidden_state
```

## Comparison: Megatron vs Other Approaches

| Approach | Type | Communication | Efficiency | Code Changes |
|----------|------|---------------|------------|--------------|
| **Megatron** | Intra-layer | All-reduce (4/layer) | 77% @ 8-way | Minimal (~few lines) |
| **GPipe** | Pipeline | Send/recv | ~65% (bubble) | Framework change |
| **Mesh-TensorFlow** | General tensor | Custom | ~80% | Compiler + DSL |
| **ZeRO** | Optimizer state | All-gather | ~90% | Library change |

**Megatron advantages:**
1. **Simple:** Just insert all-reduce ops
2. **No compiler:** Works in native PyTorch/MLX
3. **Orthogonal:** Combine with data parallelism, pipeline parallelism, ZeRO
4. **Proven:** 8.3B parameters, SOTA results on WikiText103, LAMBADA

## Expected Performance on Apple Silicon

**Configuration:** 4 × Apple M2 Ultra (8 GPUs total)

**Model:** xLSTM-large (2048 dim, 48 blocks, ~8B parameters)

**Parallelism strategy:**
- 4-way model parallel (split model across 4 GPUs)
- 2-way data parallel (2 replicas)

**Expected throughput:**

| Metric | Single GPU | 4-way Model || 4MP + 2DP |
|--------|------------|-------------|-----------|
| Model size | 2B (max) | 8B | 8B |
| Batch size | 16 | 16 | 32 |
| Throughput | 100 tok/s | 75 tok/s | 135 tok/s |
| Efficiency | 100% | 75% | **67%** |

**Reasoning:**
- Model parallel: 75% efficiency (similar to Megatron's 77%)
- Data parallel: 90% efficiency (Hogwild! on sparse gradients)
- Combined: 0.75 × 0.90 = 67.5%

**Training time reduction:**
- 8B model on single GPU: Impossible (OOM)
- 8B model with 4-way MP: ~12 hours/epoch
- 8B model with 4MP + 2DP: **~6 hours/epoch**

## Integration with MAD Framework

**MAD with model parallelism:**

```python
# mad/model/language_model.py
class MADLanguageModel(mx.nn.Module):
    def __init__(self, config):
        super().__init__()

        # Determine if model parallelism is needed
        self.model_parallel_size = config.get('model_parallel_size', 1)

        # Build layers with model parallelism
        self.layers = []
        for layer_idx, layer_spec in enumerate(config.layer_types):
            if layer_spec['type'] == 'mlstm':
                # Model parallel mLSTM
                layer = ModelParallelmLSTMBlock(
                    config,
                    model_parallel_size=self.model_parallel_size
                )
            elif layer_spec['type'] == 'swiglu':
                # Model parallel FFN
                layer = ModelParallelSwiGLU(
                    config,
                    model_parallel_size=self.model_parallel_size
                )
            # ... other layer types

            self.layers.append(layer)
```

**Benefits for MAD evaluation:**
1. **Larger models:** Evaluate 10B+ parameter architectures
2. **Faster iteration:** Train multiple large models in parallel
3. **Fair comparison:** All architectures use same computational budget

## Next Steps for Implementation

1. **MLX distributed primitives:**
   - Implement `all_reduce_mlx()` via Metal atomics
   - Support model parallel groups
   - Add communication overlapping

2. **Convert existing blocks to model parallel:**
   - `mlstm_mlx/block.py` → column/row parallelism
   - `gated_ffn_mlx/block.py` → split FFN
   - Test numerical parity (serial vs parallel)

3. **Benchmark on multi-GPU setup:**
   - Single M2 Ultra (2 GPUs) - 2-way model parallel
   - Multiple M2 Ultra - 4-8 way model parallel
   - Measure efficiency vs Megatron's numbers

4. **Integrate with Hogwild! trainer:**
   - Model parallel within each data parallel group
   - Lock-free updates across replicas
   - Expected: 20-30× speedup on 8 GPU setup

## Key Takeaways

1. **Megatron's intra-layer parallelism** is complementary to Hogwild!'s data parallelism
2. **Simple implementation:** Just a few all-reduce operations
3. **77% efficiency** at 8-way model parallel is excellent
4. **Critical for xLSTM:** Enables training 10B+ parameter models on Apple Silicon
5. **Pre-normalization matters:** Use Megatron's BERT fix for better scaling
6. **Communication optimization:** Fuse embedding loss to reduce bandwidth

## Combined Strategy: Hogwild! + Megatron

**Best of both worlds:**

```
┌─────────────────────────────────────────────┐
│  Data Parallel Group 1 (Hogwild! lock-free) │
│  ┌───────────────────────────────────────┐  │
│  │ Model Parallel Group (Megatron)       │  │
│  │ GPU 0 | GPU 1 | GPU 2 | GPU 3         │  │
│  │ [mLSTM head 0-1 | head 2-3 | ... ]    │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Data Parallel Group 2 (Hogwild! lock-free) │
│  ┌───────────────────────────────────────┐  │
│  │ Model Parallel Group (Megatron)       │  │
│  │ GPU 4 | GPU 5 | GPU 6 | GPU 7         │  │
│  │ [mLSTM head 0-1 | head 2-3 | ... ]    │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Result:** Train 10B+ xLSTM models at 20-30× speedup on multi-GPU Apple Silicon.

---

**References:**
- Megatron-LM: Shoeybi et al., 2019
- Hogwild!: Niu et al., 2011
- MLX documentation: https://ml-explore.github.io/mlx/
- Apple Metal: https://developer.apple.com/metal/

**See also:**
- `docs/HOGWILD_ANALYSIS.md` - Lock-free data parallelism
- `docs/MAD_COMPOSITION_PROPOSAL.md` - Architecture composition
- `docs/XLSTM_ARCHITECTURE_SEQUENCES.md` - Canonical patterns
