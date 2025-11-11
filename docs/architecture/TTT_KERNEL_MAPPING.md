# TTT → Kernel Mapping Reference

Quick reference for implementing Test-Time Training using existing xLSTM-metal kernels.

---

## Tent (Entropy Minimization)

**Updates**: LayerNorm parameters only (γ, β)

| Operation         | Kernel/Infrastructure                      | Location                                         |
|-------------------|--------------------------------------------|--------------------------------------------------|
| Forward pass      | Standard MLX ops                           | Native                                           |
| Entropy loss      | `mx.sum(probs * mx.log(probs))`            | Native                                           |
| Backward          | MLX autograd                               | Native                                           |
| LayerNorm compute | Multi-head LayerNorm kernel (2-4x speedup) | `kernel_development/matrix/multihead_layernorm/` |
| Parameter update  | Standard optimizer (Adam/SGD)              | Native                                           |

**Key insight**: No custom kernels needed, just freeze all parameters except LayerNorm.

---

## TTT++ (Encoder Adaptation)

**Updates**: Q/K/V projections, input gates

| Operation                    | Kernel/Infrastructure                    | Location                                                               |
|------------------------------|------------------------------------------|------------------------------------------------------------------------|
| Q/K/V forward                | `gemm_av(W, x)`                          | `kernel_development/matrix/gemm/mlx_fast_metal_kernel/gemm_kernels.py` |
| Q/K/V backward (weight grad) | `gemm_at_b(x.T, dL_dy)`                  | Same as above                                                          |
| Similarity matrix            | Standard MLX matmul (may use Metal BLAS) | Native                                                                 |
| Contrastive loss             | `mx.sum(...)` custom loss                | Native                                                                 |
| Feature alignment loss       | `mx.mean((z1.mean(0) - z2.mean(0))**2)`  | Native                                                                 |
| Backward through encoder     | MLX autograd                             | Native                                                                 |

**Performance**: GEMM kernels provide 7-10x speedup vs naive implementation.

---

## LoRA Adapters

**Updates**: Low-rank matrices A, B (rank 4-32)

| Operation                | Kernel/Infrastructure   | Location                                                               |
|--------------------------|-------------------------|------------------------------------------------------------------------|
| Adapter forward: `A × x` | `gemm_av(A, x.T)`       | `kernel_development/matrix/gemm/mlx_fast_metal_kernel/gemm_kernels.py` |
| Adapter forward: `B × z` | `gemm_av(B, z)`         | Same as above                                                          |
| Backward: `dL/dA`        | `gemm_at_b(x.T, dL_dz)` | Same as above                                                          |
| Backward: `dL/dB`        | `gemm_at_b(z.T, dL_dy)` | Same as above                                                          |
| QR init (orthogonal A)   | QR decomposition kernel | `kernel_development/matrix/qr_decomposition/`                          |
| Parameter update         | Standard optimizer      | Native                                                                 |

**Memory**: For d=2048, rank=8: 32K adapter params vs 4M frozen params (0.8% overhead).

---

## Distributed TTT (Ray)

**Updates**: Independent adapters per block partition

| Operation           | Kernel/Infrastructure                    | Location                                    |
|---------------------|------------------------------------------|---------------------------------------------|
| Block partitioning  | Ray actor creation                       | `kernel_development/` (Ray infrastructure)  |
| Per-worker GEMM     | `gemm_av`, `gemm_at_b` on each worker    | Same kernels on all workers                 |
| Gradient AllReduce  | SWAR `avgU16RoundArith` (cross-platform) | `llama.kotlin/.../SwAR.kt` (pattern)        |
| Async gradient sync | Ray `ray.get()` + optional AllReduce     | Ray infrastructure                          |
| Memory monitoring   | `xltop.py` on each worker                | `kernel_development/optimizations/xltop.py` |

**Scaling**: 32 blocks → 4 workers = 8 blocks each = ~3.75x speedup (near-perfect).

---

## QLoRA (Quantized Base + Adapters)

**Updates**: Full-precision adapters, quantized frozen base

| Operation                | Kernel/Infrastructure                   | Location                                           |
|--------------------------|-----------------------------------------|----------------------------------------------------|
| Quantize base            | `quantize(W, bits=4)`                   | `kernel_development/matrix/variable_quantization/` |
| Dequantize on-the-fly    | `dequantize(W_q, scale, zero, bits=4)`  | Same as above                                      |
| Adapter forward/backward | Same as LoRA (GEMM kernels)             | `gemm_av`, `gemm_at_b`                             |
| Gradient compression     | SWAR `avgU8RoundArith` (4x compression) | `llama.kotlin/.../SwAR.kt` (pattern)               |
| Gradient decompression   | SWAR unpack (arithmetic-only)           | Same as above                                      |

**Memory savings**: 4-bit base = 8x reduction, adapters = 0.8% overhead → ~7.5x total savings.

---

## SWAR Gradient Operations

**Cross-platform arithmetic-only operations** (no bitwise, works on MLX/PyTorch/Ray):

### 1. Gradient AllReduce (Distributed Synchronization)

**Operation**: Average gradients from N workers

| SwARBench Function | Use Case                 | Rationale                            |
|--------------------|--------------------------|--------------------------------------|
| `avgU16RoundArith` | AllReduce across workers | 16-bit precision + unbiased rounding |

**Implementation**:

```python
# Pack gradients into u16 pairs in Int32
packed_a = pack_u16_pair(grad_a[::2], grad_a[1::2])
packed_b = pack_u16_pair(grad_b[::2], grad_b[1::2])

# Average using SWAR (arithmetic-only, cross-platform)
packed_avg = avgU16RoundArith_vectorized(packed_a, packed_b)

# Unpack back to u16
grad_avg = unpack_u16_pair(packed_avg)
```

**Performance**: ~16 GB/s effective (vs ~100-200 GB/s native fp32, but 2x memory savings).

---

### 2. Gradient Momentum Accumulation (Optimizer State)

**Operation**: Exponential moving average `m_t = β * m_{t-1} + (1-β) * g_t`

| SwARBench Function | Use Case              | Rationale                                 |
|--------------------|-----------------------|-------------------------------------------|
| `avgU16TruncArith` | Momentum accumulation | Truncation acceptable for approximate EMA |

**Memory savings**: 2x (16-bit vs 32-bit fp32).

---

### 3. Gradient Compression (Network Transmission)

**Operation**: Compress for network, decompress immediately

| SwARBench Function | Use Case                 | Rationale                                    |
|--------------------|--------------------------|----------------------------------------------|
| `avgU8RoundArith`  | Extreme compression (4x) | 8-bit sufficient for transient communication |

**Bandwidth savings**: For xLSTM-7B (7B params): 28GB → 7GB per gradient transmission.

---

### 4. High-Precision Accumulation (Long TTT Sessions)

**Operation**: Prevent catastrophic cancellation over 1,000-10,000 TTT steps

| SwAR Pattern             | Use Case             | Rationale                        |
|--------------------------|----------------------|----------------------------------|
| SwAR128 (8×16-bit limbs) | Gradient accumulator | 128-bit precision prevents drift |

**Implementation**: From `SwAR128.kt` pattern:

```python
class SwAR128Accumulator:
    def __init__(self, shape):
        self.limbs = [mx.zeros(shape, dtype='uint16') for _ in range(8)]

    def accumulate(self, grad_fp32):
        grad_128 = float32_to_swar128(grad_fp32)
        for i in range(8):
            sum_val = self.limbs[i] + grad_128.limbs[i] + self.carry
            self.limbs[i] = sum_val % 65536
            self.carry = sum_val // 65536
```

**Precision**: Extends effective precision from ~10^7 ops (fp32) to ~10^15 ops (128-bit fixed-point).

---

## Optimization & Evaluation

### Hyperparameter Tuning

**Tool**: `kernel_development/optimizations/optimize_mps.py`

**TTT hyperparameters**:

- Learning rate: `(1e-5, 1e-3, 'log')`
- Adapter rank: `(4, 32, 'int')`
- Feature alignment weight: `(0.01, 1.0, 'log')`
- TTT steps: `(10, 100, 'int')`

**Usage**:

```python
from kernel_development.optimizations.optimize_mps import BayesianOptimizer

optimizer = BayesianOptimizer(
    objective=ttt_validation_accuracy,
    param_space={...}
)
best_params = optimizer.optimize(n_trials=50)
```

---

### Memory Profiling

**Tool**: `kernel_development/optimizations/xltop.py`

**Metrics**:

- MPS allocated/reserved
- RSS growth over TTT steps
- Adapter memory overhead
- Quantization savings (4-bit vs fp16)

**Usage**:

```bash
# Terminal 1: Run TTT
python run_ttt.py --model xlstm-7b --adapters lora --rank 8

# Terminal 2: Monitor
python kernel_development/optimizations/xltop.py --interval 1.0 --log ttt_memory.csv
```

---

### Output Quality Evaluation

**Tool**: `kernel_development/optimizations/judge_*.py`

**Criteria**:

- Coherence: Logical flow maintained?
- Factuality: Domain accuracy improved?
- Relevance: Focused on test distribution?

**Usage**:

```python
from kernel_development.optimizations.judge_gpt4 import LLMJudge

judge = LLMJudge(model="gpt-4")
score = judge.compare(
    prompt=test_prompt,
    output_a=base_output,
    output_b=ttt_output,
    criteria="coherence, factuality, relevance"
)
```

---

## Implementation Priority

### Phase 1: Minimal Tent TTT (1-2 days)

**Kernels needed**: None (MLX autograd + existing LayerNorm kernel)

- Freeze all params except LayerNorm
- Entropy loss + backward
- Test on distribution shift dataset

### Phase 2: LoRA Adapter TTT (3-5 days)

**Kernels needed**: GEMM (already exists), QR (already exists)

- `LoRAAdapter` class using `gemm_av`/`gemm_at_b`
- QR initialization
- Rank sweep with `optimize_mps.py`

### Phase 3: TTT++ (5-7 days)

**Kernels needed**: None (GEMM + MLX ops)

- Data augmentation
- Contrastive loss (GEMM for similarity matrix)
- Feature alignment loss
- Evaluate with `judge_*.py`

### Phase 4: Distributed TTT (7-10 days)

**Kernels needed**: None (Ray + GEMM + SWAR patterns)

- Ray actor workers
- Block partitioning
- SWAR `avgU16RoundArith` for AllReduce
- Test scaling on multi-GPU

### Phase 5: QLoRA + SWAR (5-7 days)

**Kernels needed**: None (quantization kernel + SWAR patterns)

- 4-bit base quantization
- SWAR gradient compression
- SwAR128 high-precision accumulator
- Validate numerical equivalence

---

## Zero New Metal Kernels

**All TTT requirements met by existing infrastructure:**

✅ GEMM kernels (7-10x speedup)
✅ LayerNorm kernel (2-4x speedup)
✅ QR decomposition
✅ Variable quantization
✅ SWAR patterns (arithmetic-only, cross-platform)
✅ Optimization framework (`optimize_mps.py`)
✅ Memory monitoring (`xltop.py`)
✅ Output evaluation (`judge_*.py`)
✅ Ray distributed infrastructure

**Result**: TTT implementation is **pure Python orchestration** of optimized Metal kernels.

---

## Files Referenced

### Existing Kernels

- `kernel_development/matrix/gemm/mlx_fast_metal_kernel/gemm_kernels.py` - GEMM
- `kernel_development/matrix/multihead_layernorm/` - LayerNorm
- `kernel_development/matrix/qr_decomposition/` - QR
- `kernel_development/matrix/variable_quantization/` - Quantization

### SWAR Patterns

- `llama.kotlin/external/staging/ember/src/commonMain/kotlin/ai/solace/klang/bitwise/SwAR.kt` - avgU8/u16 operations
- `llama.kotlin/external/staging/ember/src/commonMain/kotlin/ai/solace/klang/bitwise/SwAR128.kt` - High-precision
  arithmetic
- `llama.kotlin/src/nativeMain/kotlin/ai/solace/bench/SwARBench.kt` - Performance benchmarks

### Optimization Tools

- `kernel_development/optimizations/optimize_mps.py` - Hyperparameter tuning
- `kernel_development/optimizations/xltop.py` - Memory monitoring
- `kernel_development/optimizations/judge_*.py` - Output evaluation

### Documentation

- `docs/architecture/TTT_SWAR_KERNEL_SYNTHESIS.md` - Complete analysis
- `docs/architecture/KERNEL_LAB_AND_TTT_INTEGRATION.md` - Kernel infrastructure
- `docs/architecture/TTT_KERNEL_MAPPING.md` - This document
