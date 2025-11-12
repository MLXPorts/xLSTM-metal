# xLSTM-Metal Architecture

**Status:** Production (January 2025)  
**Implementation:** MLX with Metal acceleration  
**Target:** Apple Silicon (M1/M2/M3/M4)

This document describes the working architecture. For historical MAD wiring and stage-based designs, see `docs_archive/`.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│ xLSTMRunner (generate.py)                                   │
│ - Config loading, tokenizer init, generation loop           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ WiredxLSTM (models/wired_xlstm.py)                          │
│ - Embedding → Blocks → OutNorm → LMHead                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ NCPS AutoWiring (wiring/auto_wiring.py)                     │
│ - Introspects safetensors structure                         │
│ - Creates block cells (mLSTMBlock factory)                  │
│ - Sequential execution: block_0 → block_1 → ... → block_31  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ mLSTMBlock (blocks/mlstm/mlstm_block.py)                    │
│ - RMSNorm → mLSTMNeuron → Residual                          │
│ - RMSNorm → GatedFFN → Residual                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Metal Kernels (mlstm_chunkwise/)                            │
│ - Recurrent: inter-chunk state updates                      │
│ - Parallel: intra-chunk outputs                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. WiredxLSTM (Top-Level Model)

**File:** `xlstm_metal/mlx_jit/models/wired_xlstm.py`

**Responsibilities:**
- Model assembly from config
- Weight loading from safetensors
- Forward pass orchestration: embedding → blocks → head

**Key Properties:**
- Config-driven: no hardcoded layer counts
- Automatic structure discovery via NCPS AutoWiring
- Supports any xLSTM variant (1B, 7B, 13B, etc.)

**Forward Pass:**
```python
def __call__(self, input_ids, state=None):
    x = self.embedding(input_ids)           # [B, S] → [B, S, D]
    x, new_state = self.blocks(x, state)    # Sequential block execution
    x = self.out_norm(x)                    # RMSNorm
    logits = self.lm_head(x)                # [B, S, D] → [B, S, vocab]
    return logits, new_state
```

### 2. NCPS Auto-Wiring

**File:** `xlstm_metal/mlx_jit/wiring/auto_wiring.py`

**Inspiration:** Adapted from [ncps-mlx](https://github.com/MLXPorts/ncps-mlx) Neural Circuit Policy wiring patterns.

**Why NCPS?**
- **Declarative**: Structure discovered from weights, not hardcoded
- **Model-Agnostic**: Works with any xLSTM checkpoint
- **Introspectable**: Can query block types before instantiation
- **Version-Resilient**: Adapts to checkpoint structure changes

**How It Works:**
1. Read `model.safetensors.index.json`
2. Detect block count from weight keys (e.g., `backbone.blocks.0.mlstm...`)
3. Identify block types (mLSTM/sLSTM/attention) from weight patterns
4. Create factory methods for cell instantiation
5. Wire sequential connections: `block_i → block_{i+1}`

**Current Implementation:**
- **32 sequential blocks** (xLSTM-7B)
- All blocks are mLSTM type
- No parallel execution within blocks (simplicity over speed)
- State threading: each block passes (C, n, m) to next

**Code Example:**
```python
wiring = AutoWiring.from_safetensors("xlstm_7b_model")
print(f"Detected {wiring.structure['num_blocks']} blocks")

# Factory creates mLSTMBlock with correct config
block = wiring.create_block_cell(block_idx=0, config=config)
```

**Historical Note:** Earlier versions used "MAD wiring" with stage-based parallelism. This was replaced with NCPS for simplicity and reliability. See `docs_archive/components/mad/` for that design.

### 3. mLSTMBlock (Recurrent Unit)

**File:** `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_block.py`

**Architecture:**
```
Input [B, S, D]
     ↓
 ┌─────────────────┐
 │  RMSNorm        │
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │  mLSTMNeuron    │  (Q/K/V projections, gates, kernel)
 │  State: (C,n,m) │
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │  Residual Add   │  ← skip connection
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │  RMSNorm        │
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │  GatedFFN       │  (SwiGLU)
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │  Residual Add   │
 └────────┬────────┘
          ↓
Output [B, S, D]
```

**Key Parameters (xLSTM-7B):**
- `embedding_dim`: 4096
- `num_heads`: 8  
- `qk_dim`: 2048 (factor 0.5)
- `v_dim`: 4096 (factor 1.0)
- `chunk_size`: 64
- `ffn_hidden_dim`: 10944 (factor ~2.67, rounded to multiple of 64)

**State Management:**
Each mLSTMBlock maintains three tensors per head:
- **C** (covariance): `[B, NH, QK_DH, V_DH]` - outer product accumulator (k⊗v)
- **n** (normalizer): `[B, NH, QK_DH]` - denominator accumulator
- **m** (max): `[B, NH]` - running maximum for numerical stability

States are initialized to zero at sequence start, updated recurrently or chunkwise.

### 4. mLSTMNeuron (Core Computation)

**File:** `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_neuron.py`

**Components:**
1. **Projections**
   - Q, K, V: `[D, NH×head_dim]` Linear layers
   - Input gate (i): `[D, NH]`
   - Forget gate (f): `[D, NH]`

2. **Kernel** (Sequential or Chunkwise)
   - Recurrent: Simple step-by-step state updates
   - Chunkwise: Optimized parallel algorithm (current default)

3. **Output Cell**
   - Projects multi-head outputs back to model dimension
   - MultiHeadRMSNorm for per-head normalization
   - Gate-controlled output scaling

**Numerical Stability Patterns:**

Critical lessons from debugging (see `docs_archive/architecture/MLSTM_NUMERICAL_STABILITY_ANALYSIS.md`):

1. **Forget gate uses logsigmoid**: `f_log = -log(1 + exp(-f))` not `log(sigmoid(f))`
2. **Max state prevents overflow**: `m_new = max(f_log + m_old, i_gate)`, all exp relative to `m_new`
3. **Query scaling**: `q_scaled = q / sqrt(qk_dim)` applied at retrieval, key NOT scaled
4. **Denominator stabilization**: `max(|q·n|, exp(-m)) + eps` prevents division by zero
5. **State dtype**: Always float32, cast to compute dtype only for matmul

### 5. Metal Kernels (Chunkwise Algorithm)

**Files:** `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_chunkwise/`

The chunkwise algorithm achieves O(T/C + C) complexity instead of O(T²) attention or O(T) sequential:

**Phase 1 - Recurrent (Inter-Chunk):**
```
For each chunk k:
  vecB = cumsum(logsigmoid(f))     # Cumulative forget
  vecA = vecB[-1] - vecB + i       # Effective input gates
  
  # Update max state (numerical stability)
  m_new = max(vecB[-1] + m_old, max(vecA))
  
  # Exponential stabilization
  f_bar = exp(vecB[-1] + m_old - m_new)
  i_bar = exp(vecA - m_new)
  
  # Update covariance and normalizer
  C_new = f_bar * C_old + K^T @ (i_bar ⊙ V)
  n_new = f_bar * n_old + sum(i_bar ⊙ K, axis=seq)
```

**Phase 2 - Parallel (Intra-Chunk):**
```
For each position t in chunk (parallel):
  # Causal attention within chunk
  matG = Q @ K^T
  matD = exp(logsigmoid_diffs + i_gates) * qk_scale
  matS = matG ⊙ tril(matD)
  H_intra = matS @ V
  
  # Contribution from previous chunks
  H_inter = Q @ C_prev
  
  # Combine and normalize
  H_out = (H_inter + H_intra) / max(|Q·n|, exp(-m)) + eps)
```

**Metal Optimizations:**
- **Threadgroup memory**: 16×16 tiles (not 64×64 like Triton) due to 32KB limit
- **Coalesced access**: Row-major layout for unified memory efficiency  
- **Fused operations**: Gate computation + exp + accumulation in single kernel
- **Argument packing**: Pack scalars to reduce Metal buffer count (<31 limit)

See `docs_archive/porting/mlx_metal/XLSTM_MLX_ARCHITECTURE.md` lines 186-200 for detailed kernel structure.

## Data Flow

### Inference (Single Token)

```
Token ID [1]
    ↓
Embedding [1, 4096]
    ↓
┌──────────────────────┐
│ For each block (32): │
│   norm → mlstm → +   │ → state_i
│   norm → ffn → +     │
└──────────────────────┘
    ↓
OutNorm [1, 4096]
    ↓
LMHead [1, vocab_size]
    ↓
Sample next token
```

### Generation Loop

```python
state = None  # Initial state (all zeros)
tokens = [bos_token]

for step in range(max_tokens):
    logits, state = model(tokens[-1:], state)  # Incremental
    next_token = sample(logits, temperature, top_k, top_p)
    tokens.append(next_token)
    
    if next_token == eos_token:
        break
```

**Key Property:** State is threaded through generation, avoiding recomputation of past context.

## Configuration

**File:** `xlstm_7b_model/config.json`

Critical fields and their purposes:

```json
{
  "torch_dtype": "float32",              // Model weights dtype (MAIN)
  "autocast_kernel_dtype": "bfloat16",   // Selective kernel optimization
  "inference_state_dtype": "float32",    // Recurrent state storage
  "norm_reduction_force_float32": true,  // RMSNorm accumulation precision
  
  "embedding_dim": 4096,
  "num_heads": 8,
  "chunk_size": 64,
  "qk_dim_factor": 0.5,                  // Computed: qk_dim = 4096 * 0.5 = 2048
  "v_dim_factor": 1.0,                   // Computed: v_dim = 4096 * 1.0 = 4096
  "ffn_proj_factor": 2.667,              // Computed: ~10944 (rounded to 64)
  
  "gate_soft_cap": 15.0,
  "output_logit_soft_cap": 30.0,
  "eps": 1e-6
}
```

**Dtype Confusion (Bug History):**

The config has THREE dtype fields. Using the wrong one caused NaNs (see `docs_archive/COMPLETE_FIX_SUMMARY.md`):

- **WRONG**: `compute_dtype = config['autocast_kernel_dtype']` → bfloat16 everywhere → NaNs after 32 layers
- **CORRECT**: `compute_dtype = config['torch_dtype']` → float32 stable

**Lesson:** `autocast_kernel_dtype` is for selective Mixed Precision in specific ops, NOT global dtype.

## Memory Layout

**xLSTM-7B on M2 Max (64GB):**

```
Model weights:     ~14 GB  (7.16B params × fp32)
Embeddings:        ~512 MB (vocab=131072, dim=4096, fp32)
Per-block state:   ~2 MB   (C: 8×512×1024, n: 8×512, m: 8, fp32)
Total state:       ~64 MB  (32 blocks)
Activation cache:  ~128 MB (intermediate tensors, varies by batch)
──────────────────────────────────────────────────
Total:             ~15 GB  (leaves ~49GB for OS/apps)
```

**Unified Memory Advantage:**
- No CPU↔GPU copies (data stays in shared RAM)
- Metal kernels access memory directly
- Lazy evaluation reduces peak usage

## Performance Characteristics

**xLSTM-7B on M2 Max (32-core GPU):**

- **First token:** ~500ms (includes Metal shader compilation)
- **Subsequent tokens:** ~50-100ms per token
- **Throughput:** ~10-20 tokens/second (single batch)
- **Memory bandwidth:** ~200-400 GB/s (unified memory)

**Bottlenecks:**
1. **Sequential execution**: No inter-block parallelism currently
2. **Metal compilation**: First run pays JIT cost (cached after)
3. **Small batch**: Metal kernels underutilized with B=1

**Future Optimization Opportunities:**
- Multi-head parallelism (split heads across cores)
- Block-level pipelining (staged execution)
- Speculative decoding (draft models)

See `docs_archive/plan/` for experimental approaches to these.

## Design Decisions

### Why NCPS Over MAD Stages?

**MAD (Modular Architecture Design)** was the original wiring system with explicit stages and parallel block groups. Replaced because:

1. **Complexity**: Stage management, synchronization barriers
2. **Debugging**: Parallel bugs harder to isolate
3. **Parity**: Canonical implementation is sequential
4. **Performance**: Parallel gains minimal with current Metal tile sizes

NCPS auto-wiring is simpler, more maintainable, equally fast for current use case.

### Why Float32 Not BFloat16?

**Attempted:** bfloat16 for speed (half memory bandwidth)

**Result:** NaN explosions after ~10-15 blocks

**Root Cause:** bf16 has only 7-8 mantissa bits. Accumulation errors over 32 layers + 4096 dimensions → catastrophic loss of precision.

**Solution:** Float32 for everything except selective operations (future work).

See `docs_archive/COMPLETE_FIX_SUMMARY.md` and `docs_archive/FIX_DTYPE_ISSUE.md`.

### Why No Parallel Heads?

**Could implement:** Each of 8 heads on separate GPU cores

**Didn't because:**
1. Metal threadgroup overhead currently higher than compute savings
2. Head dimension (512/1024) too small to saturate cores
3. Sequential is easier to debug and validate
4. Profiling showed minimal gains (<15% speedup, not worth complexity)

Revisit when targeting larger models or multi-GPU systems.

## Testing Strategy

**Unit Tests:** Individual component validation
- `tests/test_mlstm_cell.py` - Kernel correctness
- `tests/test_rmsnorm.py` - Normalization stability
- `tests/test_wiring.py` - NCPS structure discovery

**Integration Tests:** End-to-end validation
- `tests/test_pretrained_inference.py` - Full model forward pass
- `test_numerical_parity.py` - Compare with transformers reference

**Numerical Tests:** Precision validation
- `test_parity_simple.py` - Element-wise operations
- `test_function_parity.py` - Kernel outputs vs reference

**Regression Tests:** Guard against known bugs
- NaN detection in RMSNorm (Issue: dtype confusion)
- State shape validation (Issue: k⊗v vs v⊗k)
- FFT normalization (Issue: double scaling)

See `docs/TESTING.md` (TBD) for full test suite documentation.

## Related Projects

This work builds on and shares techniques with:

- **[ncps-mlx](https://github.com/MLXPorts/ncps-mlx)** - NCPS wiring patterns for MLX
- **[m2-bert-mlx](https://github.com/MLXPorts/m2-bert-mlx)** - FFT precision, Metal kernel design
- **[ember-ml](https://github.com/SolaceHarmony/ember-ml)** - Precision enforcement (emberlint/embercoach)
- **[Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx)** - Unified memory optimization patterns

All by Sydney Renee / The Solace Project.

## References

**Original Research:**
- Beck et al., "xLSTM: Extended Long Short-Term Memory", arXiv:2405.04517
- Beck et al., "xLSTM-7B: A Recurrent LLM for Fast and Efficient Inference", arXiv:2503.13427

**MLX Framework:**
- Apple MLX Documentation: https://ml-explore.github.io/mlx/
- MLX Examples: https://github.com/ml-explore/mlx-examples

**Neural Circuit Policies:**
- Lechner et al., "Neural Circuit Policies", NeurIPS 2020
- NCPS-MLX adaptation: https://github.com/MLXPorts/ncps-mlx

---

This architecture reflects the **working, production-tested system** as of January 2025. Experimental designs are documented in `docs_archive/` for historical reference.
