# Hogwild! Analysis for xLSTM Training

**Lock-free parallel SGD and implications for MAD training**

Date: 2025-01-21
Source: Niu et al., "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent" (2011)
Relevance: Parallel training of xLSTM models on Apple Silicon multicore systems

## Executive Summary

The Hogwild! paper demonstrates that **SGD can run in parallel without locks** when the problem has **sparse gradients
**. This is directly applicable to our xLSTM training:

1. **mLSTM/sLSTM are sparse**: Each gradient update touches only a small subset of parameters
2. **Apple Silicon benefits**: 12GB/s shared memory bandwidth vs MapReduce's 10MB/s
3. **Near-linear speedup**: Theoretical guarantee when sparsity conditions hold
4. **Robust convergence**: 1/k rates with constant stepsize + exponential backoff

## Core Insight: Sparsity Enables Lock-Free Parallelism

### The Hogwild! Protocol

**Key idea:** Let processors read/write shared memory freely, even if they overwrite each other's work.

**Why it works:** When gradients are sparse (each update touches few parameters), overwrites are:

- **Rare** (low probability of collision)
- **Harmless** (small error when they occur)

### Formal Sparsity Definition

Given cost function:

```
f(x) = Σ_e∈E f_e(x_e)
```

where each `f_e` depends on small subset `e ⊂ {1,...,n}` of parameters.

**Sparsity metrics:**

- **Ω**: Max edge size `max_e |e|`
- **ρ**: Fraction of overlapping edges (density)
- **Δ**: Max variable frequency (normalized degree)

**Convergence guarantee:** Near-linear speedup when `ρ` and `Δ` are small, and number of processors `p < n^(1/4)`.

## Application to xLSTM/mLSTM

### Why xLSTM Training is Sparse

**mLSTM block structure:**

```python
# Each training example touches:
# 1. Input embedding (sparse if vocabulary >> batch size)
# 2. QKV projections (subset of weight matrix)
# 3. Gate computations (per-head parameters)
# 4. Output projection (subset of weight matrix)
```

**Sparsity sources:**

1. **Batch-level sparsity**:
    - Batch size `B` << total parameters `n`
    - Each SGD step updates parameters touched by batch
    - Example: `B=32`, `n=1e9` → `ρ ≈ 32/1e9`

2. **Sequence-level sparsity**:
    - Attention is over sequence length `S`
    - Per-head operations: `O(S * head_dim)` parameters
    - Rest of model untouched

3. **Embedding sparsity**:
    - Vocabulary size `V` (e.g., 50k)
    - Each example uses `S` tokens
    - Only `S/V` fraction of embedding weights updated
    - Example: `S=512`, `V=50k` → `Δ ≈ 0.01`

### Hypergraph Structure

**Induced hypergraph for mLSTM:**

**Nodes:** All model parameters

- Embedding weights: `V × d_model`
- QKV projections: `d_model × 3 × num_heads × head_dim`
- Gate projections: `d_model × 3 × num_heads`
- FFN weights: `d_model × ffn_dim`
- Output weights: `d_model × V`

**Hyperedges:** Each training example defines an edge `e` connecting:

- Token embeddings used in that example
- QKV parameters for attended positions
- Gate parameters for those positions
- FFN parameters (if full sequence processed)

**Sparsity analysis:**

For xlstm-large (2048 dim, 48 blocks, 50k vocab):

- Total parameters: `n ≈ 1.3 billion`
- Parameters per example: `|e| ≈ 2048 × 512 + 48 × (2048 × 512) ≈ 50M`
- **Ω ≈ 50M / 1.3B ≈ 0.04** (4% of parameters touched)
- **ρ ≈ batch_size / total_examples** (very small for large datasets)
- **Δ ≈ batch_size / total_examples** (same order as ρ)

**Conclusion:** xLSTM training satisfies Hogwild! sparsity conditions.

## Theoretical Guarantees for xLSTM

### Convergence Rate (from Proposition 4.1)

After `k` component updates with `p` processors:

```
E[f(x_k) - f*] ≤ ε
```

where `k` satisfies:

```
k ≥ (2LM²Ω(1 + 6τρ + 6τ²ΩΔ^(1/2)) log(LD₀/ε)) / (c²ϑε)
```

**Parameters:**

- `L`: Lipschitz constant of gradient
- `M`: Gradient bound
- `c`: Strong convexity modulus
- `τ`: Staleness (≈ number of processors)
- `ϑ ∈ (0,1)`: Stepsize underestimate factor

**Key insight:** When `ρ, Δ = o(1/n)` and `τ = p < n^(1/4)`:

- **Serial complexity:** `O(log(1/ε)/ε)` iterations
- **Parallel complexity:** `O(log(1/ε)/ε)` iterations (same!)
- **Speedup:** `p × speedup` (near-linear in number of processors)

### Robust 1/k Rates

The paper shows **constant stepsize + exponential backoff** achieves:

```
ε ≤ (2 log(2/β)/(1-β)) · (B/c) · 1/(k - ϑ⁻¹ log(a₀c/ϑB))
```

**Benefits over 1/k diminishing stepsize:**

- More robust to curvature misestimation
- Faster in practice (larger steps early on)
- No exponential slowdown if `c` overestimated

**Recommended protocol:**

1. Set `γ = ϑ/c` for `ϑ ∈ (0,1)`
2. Run for `K` iterations
3. Reduce `γ` by factor `β ≈ 0.37` (optimal)
4. Run for `β⁻¹K` more iterations
5. Repeat until convergence

## Implications for MAD Training

### 1. Multi-Backend Training

**Current MAD limitation:** Sequential training only

**Hogwild! enables:**

```python
# Parallel training across backends
def hogwild_train_mad(config, num_workers=8):
    model = MADLanguageModel(config)
    shared_params = model.parameters()  # Shared memory

    # Launch workers (no locks!)
    workers = [
        Worker(shared_params, dataset_shard=i)
        for i in range(num_workers)
    ]

    # Each worker:
    # 1. Sample batch
    # 2. Compute gradient
    # 3. Update shared_params directly (atomic adds)
    # 4. No synchronization!
```

**Speedup estimate:**

- Apple Silicon M2 Ultra: 24 cores (20 performance + 4 efficiency)
- Expected speedup: **10-15×** for xLSTM training
- Shared memory bandwidth: 800 GB/s (unified memory)

### 2. Heterogeneous Training (MLX + PyTorch)

**Hogwild! allows mixing backends:**

```python
# Worker 1: MLX backend (GPU)
def mlx_worker(shared_params, dataset):
    while True:
        batch = sample(dataset)
        grad = compute_grad_mlx(batch, shared_params)
        atomic_update(shared_params, grad)  # Lock-free!

# Worker 2: PyTorch backend (CPU)
def pytorch_worker(shared_params, dataset):
    while True:
        batch = sample(dataset)
        grad = compute_grad_pytorch(batch, shared_params)
        atomic_update(shared_params, grad)  # Lock-free!
```

**Key requirement:** Atomic parameter updates

- MLX: Native support via Metal atomics
- PyTorch: `torch.Tensor.add_()` is atomic on MPS

### 3. MAD Evaluation Speedup

**Current bottleneck:** Training each architecture sequentially

**With Hogwild!:**

- Train multiple architectures in parallel
- Share embedding layer across experiments (sparse updates)
- 10× speedup for architecture search

### 4. Gradient Checkpoint Sharing

**Observation:** Different layer types share gradients sparsely

```python
# Example: 7:1 pattern
# [mLSTM, FFN, mLSTM, FFN, ..., mLSTM, FFN, sLSTM, FFN]

# mLSTM and sLSTM have disjoint parameters
# Can train blocks in parallel with Hogwild!
```

## Implementation Considerations

### 1. Atomic Operations

**MLX (Metal):**

```cpp
// Metal supports atomic operations natively
kernel void atomic_add(device float* dest,
                       device float* src,
                       uint gid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit((device atomic_float*)&dest[gid],
                               src[gid],
                               memory_order_relaxed);
}
```

**PyTorch (MPS):**

```python
# PyTorch tensors support in-place atomic adds
param.add_(gradient)  # Atomic on MPS backend
```

### 2. Memory Ordering

**Challenge:** Stale gradients (processor reads old values)

**Solution:** Hogwild! is provably robust to staleness `τ < n^(1/4)`

**For xlstm-large:**

- `n ≈ 1.3B` parameters
- `n^(1/4) ≈ 190`
- Safe to use up to **190 parallel workers**
- Apple Silicon: 24 cores << 190 ✓

### 3. Stepsize Selection

**From paper:** Use constant stepsize with exponential backoff

**Recommended for xLSTM:**

```python
# Initial stepsize
γ₀ = 0.001  # Conservative estimate

# Training loop
for epoch in range(num_epochs):
    # Train for one epoch with current γ
    train_hogwild(model, dataset, stepsize=γ₀, num_workers=10)

    # Reduce stepsize
    γ₀ *= 0.37  # Optimal backoff factor from paper

    # Optional: Stop if converged
    if loss < threshold:
        break
```

### 4. Load Balancing

**Challenge:** Some examples are longer/slower than others

**Hogwild! advantage:** Natural load balancing

- Fast workers process more examples
- No synchronization barriers
- Automatically adapts to varying gradient times

## Experimental Validation Plan

### Test 1: Serial vs Hogwild! (Single Backend)

**Setup:**

- Dataset: WikiText-103 (sparse, ~100M tokens)
- Model: Small xLSTM (512 dim, 6 blocks)
- Backends: MLX only
- Workers: 1, 2, 4, 8, 16

**Metrics:**

- Wall clock time to convergence
- Final loss (should be identical)
- Speedup factor

**Expected result:** 8-10× speedup with 16 workers

### Test 2: Multi-Backend Hogwild!

**Setup:**

- Same dataset/model
- Backends: Mix of MLX (GPU) and PyTorch (CPU)
- Workers: 8 MLX + 8 PyTorch

**Metrics:**

- Convergence rate vs single-backend
- Numerical parity (outputs should match within ε)

**Expected result:** Same convergence as homogeneous, validates atomic ops

### Test 3: MAD Architecture Search

**Setup:**

- Evaluate 10 different layer sequences
- Train each for 5 epochs
- Serial vs Hogwild!

**Metrics:**

- Total wall clock time
- Per-architecture convergence

**Expected result:** 10× speedup for full search

### Test 4: Gradient Staleness

**Setup:**

- Vary staleness `τ` by controlling worker count
- Measure convergence rate vs theoretical bound

**Metrics:**

- Iterations to convergence vs `τ`
- Compare to Proposition 4.1 prediction

**Expected result:** Graceful degradation, linear up to `τ ≈ 20-30`

## Comparison to Alternatives

| Method             | Locks? | Speedup     | Complexity | Works on Apple Silicon? |
|--------------------|--------|-------------|------------|-------------------------|
| **Hogwild!**       | No     | Near-linear | Simple     | ✅ Yes (native atomics)  |
| Round-Robin        | Yes    | Sublinear   | Medium     | ⚠️ Possible (slow)      |
| MapReduce          | N/A    | Linear      | High       | ❌ No (distributed)      |
| Gradient Averaging | No     | Linear      | Medium     | ✅ Yes (but slower)      |

**Key differences:**

1. **Round-Robin (Vowpal Wabbit):**
    - Requires locks/semaphores
    - Paper shows 10× slower than Hogwild! for fast gradients
    - xLSTM gradients are fast (< 1ms) → bad fit

2. **MapReduce (Zinkevich et al.):**
    - Run `p` independent SGD instances, average output
    - Requires `p × data` passes
    - Paper shows no improvement over serial
    - Designed for clusters, not multicore

3. **Gradient Averaging (Dekel et al.):**
    - Workers compute gradients, master averages and updates
    - Requires communication overhead
    - Good for distributed, overkill for shared memory

**Conclusion:** Hogwild! is optimal for xLSTM on Apple Silicon.

## Key Takeaways

1. **xLSTM training is sparse** → Hogwild! applies directly
2. **Apple Silicon is ideal** for Hogwild! (high-bandwidth shared memory)
3. **10-15× speedup** achievable with 24 cores
4. **No code changes needed** beyond atomic ops (already supported)
5. **Robust convergence** guaranteed by theory
6. **Multi-backend training** becomes feasible (MLX + PyTorch)
7. **MAD evaluation** can be parallelized for massive speedup

## Next Steps

1. **Implement Hogwild! trainer** for MAD blocks
2. **Validate atomic operations** on MLX and PyTorch MPS
3. **Benchmark speedups** on Apple Silicon (M2 Ultra)
4. **Compare numerical parity** vs serial training
5. **Integrate with MAD evaluation** pipeline

---

**References:**

- Hogwild! paper: Niu et al., 2011 (https://arxiv.org/pdf/1106.5730)
- MAD paper: Mechanistic Architecture Design
- xlstm-large: Canonical checkpoint for validation
- Apple Metal atomics: https://developer.apple.com/metal/

**See also:**

- `docs/MAD_COMPOSITION_PROPOSAL.md` - LFM2 pattern for heterogeneous sequences
- `docs/XLSTM_ARCHITECTURE_SEQUENCES.md` - Canonical block orderings
- `mad/blocks/` - Current MAD block implementations
