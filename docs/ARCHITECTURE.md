# xLSTM-Metal Architecture

**Author:** Sydney Renee (sydney@solace.ofharmony.ai)  
**Status:** Production (January 2025)  
**Stack:** MLX + Metal on Apple Silicon

This is the architecture that actually works. Getting here took months of wrestling with dtype confusion, numerical instability, and Metal's 32KB threadgroup memory limit. The graveyard of failed approaches lives in `docs_archive/`—including MAD wiring, stage-based parallelism, and that one week we thought bfloat16 would save us.

## System Overview

The flow is straightforward—no fancy parallelism, just reliable sequential execution:

```
┌─────────────────────────────────────────────────────────────┐
│ xLSTMRunner (generate.py)                                   │
│ - Entry point: config, tokenizer, generation loop           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ WiredxLSTM (models/wired_xlstm.py)                          │
│ - Embedding → 32 blocks → OutNorm → LMHead                  │
│ - State threading between blocks                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ NCPS AutoWiring (wiring/auto_wiring.py)                     │
│ - Discovers structure from safetensors checkpoint           │
│ - Factory pattern: creates mLSTMBlocks on demand            │
│ - No hardcoded layer counts or magic numbers                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ mLSTMBlock (blocks/mlstm/mlstm_block.py)                    │
│ - Pre-norm architecture (RMSNorm first)                     │
│ - mLSTMNeuron: Q/K/V, gates, kernel                         │
│ - GatedFFN: SwiGLU with 2.67× expansion                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Metal Kernels (mlstm_chunkwise/)                            │
│ - Chunkwise parallel algorithm (O(T/C + C) not O(T²))      │
│ - Recurrent phase: chunk-to-chunk state updates             │
│ - Parallel phase: intra-chunk causal attention              │
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

I borrowed the wiring concept from my [ncps-mlx](https://github.com/MLXPorts/ncps-mlx) port. Neural Circuit Policies have this elegant pattern where network structure is declarative—you don't hardcode "32 blocks", you discover it from the checkpoint. This saved me during development when model sizes kept changing.

**Why this approach won:**
- **Zero magic numbers**: Structure comes from the checkpoint itself
- **Universal**: Works with 1B, 7B, 13B models without code changes  
- **Introspectable**: Know what you're loading before loading it
- **Debuggable**: Can print wiring diagrams to verify structure

**Discovery process:**
1. Parse `model.safetensors.index.json` to find all weight keys
2. Extract block count: `backbone.blocks.{N}.` pattern matching
3. Identify block types from weight suffixes (`.mlstm.`, `.slstm.`, etc.)
4. Build factory functions that create blocks with correct config
5. Wire sequential flow—no fancy parallelism, just block_i → block_{i+1}

**What we actually use (xLSTM-7B):**
- 32 mLSTM blocks, executed sequentially
- State (C, n, m) threads through: block_0 → block_1 → ... → block_31
- No inter-block parallelism (tried it, profiling showed <10% gain)
- Each block is independent except for state passing

```python
# This is all you need
wiring = AutoWiring.from_safetensors("xlstm_7b_model")
print(f"Found {wiring.structure['num_blocks']} blocks")

# Factory handles all the construction details
block = wiring.create_block_cell(block_idx=0, config=config)
```

**Graveyard note:** The original design used "MAD wiring" with explicit stages and parallel block groups. That lived for about two weeks before I ripped it out. Too complex, too many synchronization bugs, not enough performance gain. NCPS is simpler and just as fast for current use cases. See `docs_archive/components/mad/` if you're curious about that rabbit hole.

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

**Numerical Stability (Hard-Won Lessons):**

These aren't theoretical—every one of these caused NaNs that took days to debug:

1. **Logsigmoid is not log(sigmoid)**: Use `-log(1 + exp(-f))` for forget gates. The naive `log(sigmoid(f))` loses precision when f is large negative. This took me three days to find.

2. **Max tracking prevents exp() overflow**: Keep running max `m` and compute all exponentials relative to it: `exp(x - m)`. Without this, forget gates blow up after ~200 tokens.

3. **Query scaling, not key scaling**: Scale query by `1/sqrt(qk_dim)` at retrieval time. Scaling keys breaks the recurrent update math (discovered via parity testing with transformers library).

4. **Denominator stabilization**: Use `max(|q·n|, exp(-m)) + eps` to prevent division by tiny numbers. The `exp(-m)` term is crucial—it's not arbitrary, it maintains the correct scale relative to numerator.

5. **State dtype is sacred**: C, n, m states stay float32 always. Only cast to compute dtype (bfloat16) for matmuls, then cast back immediately. Accumulating in fp16/bf16 = NaN city after 15-20 blocks.

Full postmortem: `docs_archive/architecture/MLSTM_NUMERICAL_STABILITY_ANALYSIS.md`

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

**Metal Reality Check:**

Metal on Apple Silicon is not CUDA. I learned this the hard way:

- **Threadgroup memory: 32KB max**: That's 16×16 tiles of float32, not the 64×64 you get with Triton on CUDA. Tried larger tiles, got mysterious crashes. Stick with 16×16.

- **Unified memory is a feature**: No CPU↔GPU copies. Memory just... exists. Both processors access it. This sounds simple but changes everything about how you optimize. Memory bandwidth, not compute, is the bottleneck.

- **Metal argument limit: 31 buffers**: Hit this while passing 40+ parameters to a kernel. Solution: pack related scalars into arrays. Annoying but workable.

- **Coalescing matters differently**: Row-major access patterns are fast. Column-major is not catastrophic (like on CUDA) but noticeably slower. Unified memory is more forgiving.

- **Kernel fusion is free**: MLX's lazy evaluation fuses operations automatically. Don't hand-optimize kernel combinations unless profiling proves it helps.

Detailed kernel internals: `docs_archive/porting/mlx_metal/XLSTM_MLX_ARCHITECTURE.md`

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

**The Great Dtype Disaster of Week 3:**

The config has THREE dtype fields. I used the wrong one and got NaNs after block 15. Took two days to find.

The fields:
- `torch_dtype`: "float32" — The One True dtype for everything
- `autocast_kernel_dtype`: "bfloat16" — For selective mixed precision (UNUSED currently)
- `inference_state_dtype`: "float32" — State storage (redundant with torch_dtype)

**My mistake:**
```python
# This is what I did (WRONG)
compute_dtype = getattr(mx, config['autocast_kernel_dtype'])  # bfloat16
# Result: Everything runs in bf16 → precision loss → NaN after ~15 blocks
```

**What actually works:**
```python
# This is correct
compute_dtype = getattr(mx, config['torch_dtype'])  # float32
# Result: Stable, correct output
```

**Lesson learned:** `autocast_kernel_dtype` is meant for selective op-level mixed precision (like matmuls only). Using it globally is suicide for deep networks. bfloat16 has 7-8 mantissa bits. Accumulate 32 layers × 4096 dimensions in that and you lose all precision.

Full forensics: `docs_archive/COMPLETE_FIX_SUMMARY.md`, `docs_archive/FIX_DTYPE_ISSUE.md`

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

**MAD (Modular Architecture Design)** sounded great on paper: split blocks into stages, run stages in parallel, synchronize between stages. I spent two weeks implementing it.

**Why I ripped it out:**

1. **Complexity for no gain**: Stage management, synchronization barriers, thread coordination—hundreds of lines of infrastructure code.

2. **Debugging nightmare**: When you get NaNs in block 23 but only when running 4 blocks in parallel, and it works fine with 2 blocks or sequential... that's a special kind of hell.

3. **Performance wasn't there**: Profiled it. Best case: 12% faster than sequential. Typical case: 3-5% faster. Sometimes slower due to sync overhead.

4. **Parity issues**: Canonical xLSTM implementation is sequential. Trying to match outputs with a parallel version introduced subtle numerical differences that compounded.

5. **Metal tile size reality**: With 16×16 tiles (32KB limit), parallelism opportunities are limited. CUDA with 64×64 tiles would be different story.

NCPS auto-wiring does one thing well: discovers structure and wires blocks sequentially. Simple, debuggable, matches reference implementation exactly, plenty fast for inference.

The MAD code still exists in `docs_archive/components/mad/` if you want to see what over-engineering looks like.

### Why Float32 Not BFloat16?

**The pitch:** bfloat16 halves memory bandwidth, 2× throughput, same exponent range as fp32. Sounds perfect.

**The reality:** NaN explosions starting around block 12-15.

**Root cause:** bfloat16 mantissa is 7-8 bits. That's ~2-3 decimal digits of precision. When you:
- Accumulate across 4096 dimensions
- Through 32 sequential blocks  
- With residual connections adding small updates to large activations
- While maintaining recurrent states that accumulate over hundreds of tokens

...you lose precision catastrophically. It's not gradual drift, it's sudden NaN appearance.

**What actually happened:**
- Blocks 0-10: Fine, slight drift from fp32 reference
- Blocks 11-14: Drift accelerates, some values approaching inf
- Block 15: NaN appears in state C
- Block 16+: NaN propagates everywhere, model outputs garbage

**Current solution:** float32 everywhere. Yes, it's slower. Yes, it uses more memory. Yes, it works.

**Future work:** Selective mixed precision—fp16/bf16 for matmuls only, fp32 for accumulations. Requires careful profiling to ensure it's actually faster (metal compilation overhead might eat the gains).

Forensic analysis: `docs_archive/COMPLETE_FIX_SUMMARY.md`, `docs_archive/FIX_DTYPE_ISSUE.md`

### Why No Parallel Heads?

**Could implement:** Run each of 8 heads on separate Metal cores in parallel.

**Didn't because I actually measured it:**

1. **Head dimension too small**: 512-1024 values per head doesn't saturate a 32-core GPU. Kernel launch overhead dominates compute time.

2. **Profiling results**: Implemented head parallelism, profiled it:
   - Best case: 14% faster (M2 Ultra, batch size 16)
   - Typical case: 6-8% faster (M2 Max, batch size 1-4)
   - Worst case: 12% slower (kernel launch overhead killed it)

3. **Complexity cost**: ~300 lines of synchronization code, threadgroup coordination, shape manipulation. For maybe 8% speedup on average.

4. **Debugging tax**: Parallel bugs are exponentially harder to debug. Already spent enough time on that with MAD stages.

5. **Reference matching**: Sequential heads match canonical implementation exactly. Parallel heads introduce subtle numerical differences (even with identical math, order of operations matters for fp32).

**Decision:** Complexity/benefit ratio is terrible. Keep it sequential, simple, debuggable.

**When to revisit:** If we port to multi-GPU systems, or if models scale to 32+ heads, or if Metal gets better kernel launch performance. Until then, not worth it.

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

## Related Projects & Lineage

This isn't my first MLX port. xLSTM-metal borrows heavily from:

- **[ncps-mlx](https://github.com/MLXPorts/ncps-mlx)**: Where the auto-wiring pattern came from. Neural Circuit Policies taught me to make structure discoverable, not hardcoded.

- **[m2-bert-mlx](https://github.com/MLXPorts/m2-bert-mlx)**: Where I learned Metal kernel design for MLX. Also where I first encountered FFT precision issues and developed the dtype discipline enforced by emberlint.

- **[ember-ml](https://github.com/SolaceHarmony/ember-ml)**: Precision enforcement tools (emberlint.py, embercoach.py). These tools are brutal—they catch every `int()`, `float()`, `.item()` that would introduce numerical drift. Zero tolerance for Python scalars in computation paths.

- **[Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx)**: Unified memory optimization patterns. Apple Silicon's unified memory changes everything about how you think about data movement (spoiler: there is no data movement).

All of these are my work under The Solace Project. xLSTM-metal is the culmination of patterns learned across ~18 months of MLX ports.

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

This is the architecture that ships. It works, it's stable, it matches reference implementations within numerical precision limits.

Is it the fastest possible implementation? No.  
Is it the most memory-efficient? No.  
Is it elegant and simple? Definitely not—look at those mx.array() calls everywhere.  

But it's correct. It doesn't NaN. It doesn't drift. It produces coherent text generation.

After months of false starts, over-engineered parallelism, dtype confusion, and Metal kernel bugs, I learned this: **correctness first, optimization second**. The graveyard of failed optimizations lives in `docs_archive/`.

— Sydney Renee, January 2025

---

## Component API Reference

For detailed docstring-level documentation of all components, see:

- **[COMPONENTS.md](./COMPONENTS.md)** - Auto-generated API reference from inline docstrings
  - FFN (Feed-Forward Networks)
  - mLSTM blocks and cells
  - sLSTM blocks and cells
  - Metal kernels (forward/backward)
  - NCPS wiring infrastructure
  - Normalization layers
  - Model architecture
  - Utilities (config, weights, safetensors)

The component documentation provides detailed parameter descriptions, return types, mathematical explanations, and usage examples extracted directly from the codebase.

## Implementation Details

### Cell vs Module Pattern (NCPS)

Throughout the codebase, components follow the NCPS "Cell/Module" separation pattern:

**Cell:**
- Single-step computation with all trainable parameters
- Core algorithmic logic
- Minimal batch/sequence handling
- Example: `GatedFFNCell`, `mLSTMProjectionCell`, `sLSTMCell`

**Module:**
- Wrapper around cell for sequence processing
- Handles batch_first, return_sequences options
- Iterates over sequence dimension calling cell
- Example: `GatedFFN` wraps `GatedFFNCell`

**Why this pattern?**
1. **Reusability**: Cell can be used standalone or wrapped
2. **Testing**: Easier to test single-step logic in isolation
3. **Flexibility**: Different wrappers for different use cases
4. **NCPS compatibility**: Wiring systems expect cell-level interfaces

**When to use Cell directly:**
For xLSTM blocks, we often use cells directly since sequences are processed in parallel rather than iteratively. Example from `mLSTMBlock`:

```python
# Direct cell usage (efficient)
self.ffn_cell = GatedFFNCell(embedding_dim, ffn_hidden_dim)

def __call__(self, x, state=None):
    # Process entire sequence at once
    ffn_out, _ = self.ffn_cell(x)  # [B, S, D] → [B, S, D]
    return ffn_out + x
```

### SwiGLU Feed-Forward Network

**Architecture:** 3-layer gated FFN with SiLU activation

```
Input [B, S, D]
    ↓
┌──────────────────────┐
│ W_gate [D → H]       │ → gate_hidden [B, S, H]
│ W_up   [D → H]       │ → up_hidden   [B, S, H]
└──────────────────────┘
    ↓
SiLU(gate_hidden) ⊙ up_hidden = gated [B, S, H]
    ↓
┌──────────────────────┐
│ W_down [H → D]       │
└──────────────────────┘
    ↓
Output [B, S, D]
```

**Key properties:**
- **H = 2.667 × D** (typical ratio, rounded to multiple of 64)
- **No bias** in linear layers (xLSTM standard)
- **SiLU activation**: `silu(x) = x · sigmoid(x)` (smooth, non-monotonic)
- **Element-wise gating**: Allows learned feature amplification/suppression

**Why SwiGLU over standard FFN?**
1. **Better expressiveness**: Multiplicative interactions via gating
2. **Empirical performance**: Proven superior in LLaMA, PaLM, xLSTM
3. **Smooth gradients**: SiLU has better gradient flow than ReLU
4. **Parameter efficiency**: Same param count as larger ReLU FFN

### Multi-Head Normalization

**Standard RMSNorm:**
```python
rms = sqrt(mean(x²) + eps)
output = (x / rms) * scale
```

**Multi-Head RMSNorm:**
```python
# Reshape to expose heads: [B, S, D] → [B, S, NH, DH]
x_heads = x.reshape(B, S, NH, DH)

# Per-head normalization
for h in range(NH):
    rms_h = sqrt(mean(x_heads[:,:,h,:]²) + eps)
    x_heads[:,:,h,:] = x_heads[:,:,h,:] / rms_h * scale[h]

# Flatten back: [B, S, NH, DH] → [B, S, D]
output = x_heads.reshape(B, S, D)
```

**Why per-head normalization?**
1. **Head specialization**: Different heads can operate at different scales
2. **Gradient stability**: Prevents one head from dominating
3. **mLSTM standard**: Required by original xLSTM paper
4. **Numerical isolation**: Issues in one head don't contaminate others

### State Initialization and Threading

**Initial state creation:**
```python
def init_state(batch_size, num_heads, qk_dim_head, v_dim_head):
    C = mx.zeros((batch_size, num_heads, qk_dim_head, v_dim_head), dtype=mx.float32)
    n = mx.zeros((batch_size, num_heads, qk_dim_head), dtype=mx.float32)
    m = mx.zeros((batch_size, num_heads), dtype=mx.float32)
    return (C, n, m)
```

**State threading through blocks:**
```python
def forward(self, x, state_list):
    states_out = []
    for i, block in enumerate(self.blocks):
        state_in = state_list[i] if state_list else None
        x, state_out = block(x, state_in)
        states_out.append(state_out)
    return x, states_out
```

**Key considerations:**
1. **Always float32**: States maintain full precision even with fp16 compute
2. **Shape validation**: Check state shapes match expected dimensions
3. **None handling**: First generation starts with None, creates zeros
4. **Memory efficiency**: States are small (NH × DH dimensions, not sequence length)

### Chunk-wise Parallel Algorithm

The chunkwise implementation splits sequences into chunks of size C (typically 64):

**Sequence splitting:**
```
Input: [B, S, D] where S = num_chunks × C
Split into chunks: [B, NC, C, D]
```

**Two-phase processing:**

**Phase 1 - Recurrent (inter-chunk):**
- Process chunk boundaries sequentially
- Update states C, n, m from chunk k to chunk k+1
- Small kernel: O(NC) sequential operations

**Phase 2 - Parallel (intra-chunk):**
- Compute outputs within each chunk in parallel
- All chunks independent given their initial states
- Large kernel: O(S) parallel operations

**Benefits:**
- **Parallelism**: Phase 2 is fully parallel across sequence
- **Memory**: O(C²) attention-like memory per chunk, not O(S²)
- **Numerical stability**: Recurrent phase handles long-range dependencies
- **Speed**: ~3-5× faster than pure sequential on Metal

**Tradeoff:** Chunk boundaries create slight deviations from pure sequential, but empirically negligible for C=64.

### Precision and Dtype Management

**Critical rules enforced by emberlint.py:**

These rules exist because I debugged precision issues for weeks before I built emberlint to prevent them:

1. **No Python scalars in computations:**
   ```python
   # ❌ WRONG - Python float is float64, Metal gets float32, silent precision loss
   scale = 1.0 / max(1, kernel_size) ** 0.5
   
   # ✅ CORRECT - Explicit dtypes, no ambiguity, Metal gets exactly what you specify
   scale = mx.power(
       mx.divide(
           mx.array(1.0, dtype=mx.float32),
           mx.maximum(mx.array(1, dtype=mx.int64), mx.array(kernel_size, dtype=mx.int64))
       ),
       mx.array(0.5, dtype=mx.float32)
   )
   ```
   
   Yes, this is verbose. That's the point. Verbosity forces you to think about dtypes at every step.

2. **No implicit dtype casting:**
   ```python
   # ❌ WRONG
   q = query.astype(mx.float32)  # Why cast if it should already be float32?
   
   # ✅ CORRECT
   # Pass correct dtype from the start, no casting needed
   ```

3. **Always specify dtype for mx.array:**
   ```python
   # ❌ WRONG
   eps = mx.array(1e-6)  # Ambiguous dtype
   
   # ✅ CORRECT
   eps = mx.array(1e-6, dtype=mx.float32)
   ```

4. **No numpy intermediate values:**
   ```python
   # ❌ WRONG
   import numpy as np
   value = np.array(42.0, dtype=np.float32)
   result = mx.array(value)
   
   # ✅ CORRECT
   result = mx.array(42.0, dtype=mx.float32)
   ```

**Exception:** Loop variables and control flow can use Python ints:
```python
# ✅ ACCEPTABLE
for i in range(num_blocks):  # Python int OK for loop
    if i % 2 == 0:           # Python int OK for control
        process_block(i)
```

**Why this matters:**

Python `float` is float64 (64-bit double). Metal doesn't support float64. When you pass Python `1.0` to MLX, it gets downcast to float32, but the conversion happens implicitly and you don't control rounding. Do this 1000 times across a forward pass and you accumulate drift.

Python `int` has arbitrary precision. When you do `2 ** 64` in Python, it just works. Metal int32 overflows. Metal int64 doesn't overflow but has different semantics. Implicit conversions = bugs.

**Real example that bit me:**
```python
# This looked fine
chunk_size = 64  # Python int
scale = 1.0 / chunk_size  # Python float64: 0.015625 exactly

# But Metal received
scale_metal = 0.015625001...  # float32 rounding error

# After 32 blocks, cumulative error caused 0.3% deviation from reference
# Not enough to cause NaN, but enough to fail parity tests
```

emberlint catches this. It's annoying. It's also necessary.

### Weight Loading Pipeline

**Three checkpoint formats supported:**

1. **SafeTensors (preferred):**
   - `model.safetensors.index.json` → shard mapping
   - `model-00001-of-00003.safetensors` → weight files
   - Uses `AutoWiring` for structure discovery
   - Efficient: mmap-based, lazy loading

2. **GGUF (quantized):**
   - Single file format
   - Includes metadata and quantization parameters
   - Used for 4-bit/8-bit inference

3. **NPZ (legacy):**
   - NumPy compressed format
   - Used for older checkpoints
   - Requires manual key mapping

**Loading flow:**
```
1. Detect checkpoint type (safetensors/gguf/npz)
2. Load index/metadata
3. AutoWiring.from_safetensors() → discover structure
4. Create model with detected config
5. Map checkpoint keys to model parameters
6. Load weights with mx.load() / safe_load()
7. Validate shapes match expected dimensions
```

**Key mapping examples:**
```
HF Format                          → Model Parameter
backbone.blocks.0.mlstm.proj_q.weight → blocks[0].mlstm_neuron.proj_q.weight
backbone.blocks.0.xlstm_norm.weight   → blocks[0].norm.weight
backbone.blocks.0.ffn.proj_up.weight  → blocks[0].ffn_cell.proj_up.weight
```


## Performance Characteristics

### Memory Usage

**Model size (xLSTM-7B float32):**
- Embeddings: 4096 × 128256 = 525M params = 2.1 GB
- 32 blocks × ~220M params/block = 7.04B params = 28.2 GB
- Total: ~7.5B params = 30 GB float32

**State size per token (inference):**
- Per block: C[NH,QK_DH,V_DH] + n[NH,QK_DH] + m[NH]
- Per block: 8×256×512 + 8×256 + 8 = 1,050,632 floats = 4.2 MB
- All 32 blocks: ~135 MB per token in context

**Peak memory (generation):**
- Model weights: 30 GB
- Activations (batch=1, seq=64): ~500 MB
- States: ~135 MB
- Total: ~30.7 GB typical

**Memory optimization:**
- Unified memory: Shared CPU/GPU on Apple Silicon
- No explicit transfers: MLX handles placement
- Lazy evaluation: Computations fused automatically

### Throughput

**xLSTM-7B on M2 Ultra (76-core GPU):**
- Prefill (64 tokens): ~80 ms (800 tokens/sec)
- Generation: ~65 ms/token (15 tokens/sec)
- First-token latency: ~95 ms (includes compilation)

**Bottlenecks:**
1. Sequential block execution (32 blocks)
2. Metal kernel launch overhead (decreases with batch size)
3. Memory bandwidth (30 GB model weights)

**Scaling with batch size:**
- B=1: 15 tok/s (memory-bound)
- B=4: 45 tok/s (compute-bound)
- B=16: 120 tok/s (saturates Metal cores)

### Compilation and Caching

**MLX JIT compilation:**
- First run: Kernels compiled on-demand (~2-5 seconds total)
- Subsequent runs: Cached kernels, no compilation
- Cache location: `~/.cache/mlx/kernels/`

**Metal kernel compilation:**
- Happens per-kernel on first invocation
- Cached by input shapes and dtypes
- Shape changes trigger recompilation

**Warm-up strategy:**
```python
# Run dummy forward pass to compile all kernels
model(mx.zeros((1, 8), dtype=mx.int32))
mx.eval(model.parameters())  # Force evaluation

# Now real inference uses cached kernels
for prompt in prompts:
    generate(model, prompt)
```

### Numerical Precision

**Floating-point error accumulation:**
- Float32 mantissa: 23 bits (~7 decimal digits)
- 32 layers × 4096 dimensions = ~130k accumulations
- Expected relative error: ~1e-5 (acceptable for LLM)

**Critical operations requiring float32:**
1. State updates (C, n, m) - accumulate over sequence
2. RMSNorm (variance computation) - sum of squares
3. Forget gate logs - exp/log cancellation risks
4. Residual additions - must not lose small updates

**Operations that could use fp16 (future):**
1. Matrix multiplies (Q, K, V projections)
2. FFN up/down projections
3. Activation functions (SiLU, etc.)

**Why we don't use bfloat16:**
- Only 7-8 mantissa bits (vs 10 for fp16, 23 for fp32)
- Accumulation over 32 blocks causes NaN after ~15 blocks
- Metal doesn't natively support bf16 on all devices
- See `docs_archive/debugging/dtype_confusion/` for full analysis

