# MAD Parallelism Analysis

## Executive Summary

After analyzing MAD configurations and implementations, I found **two types of parallelism** in the MAD framework:

1. **Mixture of Experts (MoE) Parallelism** - Multiple experts computed in parallel with routing
2. **Multi-head Parallelism** - Attention/Hyena with parallel heads
3. **Algorithm Parallelism** - Choice between "quadratic" vs "linear" attention implementations

**Key Finding:** The parallelism is **within layers**, not **between blocks**. Blocks still execute sequentially.

---

## 1. Mixture of Experts (MoE) Parallelism

### Configuration Examples

```yaml
# configs/layers/moe-mlp.yml
dim: 128
dim_inner: 16
num_experts: 8        # Total number of expert networks
active_experts: 2     # Number of experts active per token
bias: false
```

```yaml
# configs/layers/hyena-experts.yml
dim: 128
dim_inner: 16
num_experts: 8
active_experts: 2
order: 2
num_heads: 1
# ... other hyena params
```

### Implementation Pattern

From `mad/model/layers/mlp.py` (lines 112-170):

```python
class MoeMlp(nn.Module):
    """
    Mixture of Experts (MoE) GLU.

    Key Features:
    - num_experts: Total expert capacity
    - active_experts: Sparse activation (top-k routing)
    - Router network selects which experts process each token
    """

    def __init__(self, dim, num_experts, active_experts, ...):
        super().__init__()

        # Projections scaled by num_experts
        self.up1 = nn.Linear(dim, dim_inner * num_experts, bias=bias)
        self.up2 = nn.Linear(dim, dim_inner * num_experts, bias=bias)
        self.down1 = nn.Linear(dim_inner * num_experts, dim * num_experts, bias=bias)

        # Router: learns which experts to activate
        self.router = nn.Linear(dim, num_experts)

        self.num_experts = num_experts
        self.active_experts = active_experts

    def forward(self, x):
        # Compute all experts in parallel
        x1, x2 = self.up1(x), self.up2(x)
        z = self.down1(x1 * self.act(x2))

        # Router scores: which experts to use?
        scores = self.router(x)  # [b, l, num_experts]

        # Top-k routing: only keep top active_experts
        topk_scores = scores.topk(self.active_experts, dim=-1)[0]
        min_score = topk_scores.min(dim=-1, keepdim=True)[0]
        scores = torch.where(scores < min_score,
                            torch.zeros_like(scores),
                            scores)
        scores = F.softmax(scores, dim=-1)

        # Mix expert outputs
        z = z.reshape(z.shape[0], z.shape[1], self.dim, self.num_experts)
        return (z * scores.unsqueeze(-2)).sum(dim=-1)
```

### Key Insights

1. **All experts computed in parallel** - Single large matrix multiply instead of `num_experts` separate ones
2. **Sparse activation** - Only `active_experts` contribute to output (via router)
3. **Learned routing** - Router network learns which experts are best for each input
4. **GPU-efficient** - Parallelism happens via batched matrix operations

### Usage in Architecture

From `scripts/architecture_improvement.py` (line 83):

```python
# Example model with MoE
model_layers = ['mh-hyena', 'moe-mlp', 'mh-attention', 'moe-mlp']
```

This creates:
```
Block 0: Multi-head Hyena
Block 1: MoE MLP (8 experts, top-2 routing)
Block 2: Multi-head Attention
Block 3: MoE MLP (8 experts, top-2 routing)
```

**Sequential block execution** with **parallel experts within MoE blocks**.

---

## 2. Multi-Head Parallelism

### Configuration Examples

```yaml
# configs/layers/mh-attention.yml
dim: 128
num_heads: 16        # Parallel attention heads
head_dim: 64
causal: true
use_flash_attn: true
```

```yaml
# configs/layers/mh-hyena.yml
dim: 128
num_heads: 4         # Parallel Hyena heads
order: 2
# ...
```

### Implementation Pattern

Multi-head attention naturally parallelizes across heads:

```python
# Simplified multi-head attention
query = query.view(B, S, num_heads, head_dim).transpose(1, 2)  # [B, NH, S, DH]
key = key.view(B, S, num_heads, head_dim).transpose(1, 2)
value = value.view(B, S, num_heads, head_dim).transpose(1, 2)

# Each head computed in parallel (GPU batch dimension)
attn = torch.matmul(query, key.transpose(-2, -1))  # [B, NH, S, S]
attn = F.softmax(attn, dim=-1)
out = torch.matmul(attn, value)  # [B, NH, S, DH]

# Concatenate heads
out = out.transpose(1, 2).reshape(B, S, dim)
```

**GPU parallelism:** All heads computed simultaneously via batched operations.

---

## 3. Algorithm Parallelism (Linear Attention)

### Configuration

```yaml
# configs/layers/linear-attention.yml
dim: 128
feature_map: elementwise_product
parallel_implementation: quadratic   # or "linear"
```

### Implementation from `attention_linear.py`

```python
def parallel_forward(self, x, q, k, v):
    if self.parallel_implementation == "quadratic" or causal_dot_product is None:
        # Materialize full attention matrix
        A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k)  # [B, H, N, M]
        A_qk = torch.tril(A_qk)  # Causal mask
        y = torch.einsum("bhnm,bhme->bhne", A_qk, v)
        # ... normalization

    elif self.parallel_implementation == "linear" and causal_dot_product is not None:
        # Use optimized kernel (avoids materializing attention matrix)
        v = causal_dot_product(
            q.contiguous().to(dtype=torch.float32),
            k.contiguous().to(dtype=torch.float32),
            v.contiguous().to(dtype=torch.float32)
        )
        # ... normalization
```

**This is NOT block-to-block parallelism!** It's choosing between:
- **"quadratic"**: O(n²) memory, easier to parallelize on GPU
- **"linear"**: O(n) memory, uses specialized CUDA kernel

Both run **within a single layer** during forward pass.

---

## 4. Hyena Experts (Combined MoE + Hyena)

### Configuration

```yaml
# configs/layers/hyena-experts.yml
num_experts: 8
active_experts: 2
order: 2
num_heads: 1
```

### Implementation Highlights

From `hyena.py` (lines 359-475):

```python
class HyenaExpertsOperator(HyenaOperator):
    def __init__(self, num_experts=8, active_experts=2, ...):
        super().__init__(...)

        # Scale projections by num_experts
        self.in_proj = nn.Linear(
            self.dim,
            (order + 1) * self.dim_inner * self.num_experts
        )

        # Router for expert selection
        self.router = nn.Linear(self.dim, self.num_experts)

    def forward(self, u):
        # Project input (all experts in parallel)
        u = self.in_proj(u)

        # Apply short filter
        uc = self.short_filter(u)

        # ... Hyena computation with long convolutions ...

        # Router: select top-k experts
        scores = self.router(u_pre)
        topk_scores = scores.topk(self.active_experts, dim=-1)[0]
        min_score = topk_scores.min(dim=-1, keepdim=True)[0]
        scores = torch.where(scores < min_score,
                            torch.zeros_like(scores),
                            scores)
        scores = F.softmax(scores, dim=-1)

        # Mix expert outputs
        y = y.reshape(y.shape[0], y.shape[1], self.dim_inner, self.num_experts)
        y = (y * scores.unsqueeze(-2)).sum(dim=-1)

        return self.out_proj(y)
```

**Combines:**
- Hyena's efficient long-range modeling
- MoE's sparse expert capacity
- All computed in parallel via batched operations

---

## Architecture Patterns in MAD

### Example from `architecture_improvement.py`

```python
# Line 75-84: Model definitions
{
    # Base model: Sequential hyena + MLP
    'Hyena + SwiGLU':
        ['hyena', 'swiglu', 'hyena', 'swiglu'],

    # + Multi-head parallelism
    'MH Hyena + SwiGLU':
        ['mh-hyena', 'swiglu', 'mh-hyena', 'swiglu'],

    # ++ Heterogeneous (attention mixed in)
    'Striped MH Hyena + SwiGLU':
        ['mh-hyena', 'swiglu', 'mh-attention', 'swiglu'],

    # +++ MoE expert parallelism
    'Striped MH Hyena + MoE':
        ['mh-hyena', 'moe-mlp', 'mh-attention', 'moe-mlp']
}
```

### Architecture Composition

```python
def make_model_fn(task, vocab_size, max_length):
    # Get layer modules from registry
    layers = [layer_registry[l]['module'] for l in model_layers]
    layer_configs = [load_yml(layer_registry[l]['cfg']) for l in model_layers]

    # Update configs
    for layer_config in layer_configs:
        layer_config['max_length'] = max_length

    # Create model (sequential blocks with residuals)
    if task == 'compression':
        model = AutoEncoder(...)
    else:
        model = LanguageModel(
            layers=layers,
            layer_cfgs=layer_configs,
            vocab_size=vocab_size,
            ...
        )

    return model
```

**Result:** Sequential blocks, but with parallel computation **within** MoE/multi-head blocks.

---

## Summary: Types of Parallelism in MAD

| Type | Level | Example | Implementation |
|------|-------|---------|----------------|
| **MoE Experts** | Within layer | 8 experts, top-2 routing | Batched matmul, sparse activation |
| **Multi-head** | Within layer | 16 attention/hyena heads | Batched operations across heads |
| **Algorithm choice** | Within layer | "quadratic" vs "linear" attention | Different memory/compute tradeoffs |
| **GPU parallelism** | Hardware | All of the above | CUDA kernels, tensor parallelism |

**What MAD does NOT have:**
- ❌ Block-to-block parallelism (still sequential)
- ❌ Threading/async (Python GIL limitations)
- ❌ Model parallelism (different from data parallelism)

---

## Implications for xLSTM-MAD-NCPS

### What We Can Adopt from MAD

1. **MoE Blocks**
   ```python
   # Future xLSTM variant
   layer_types = [
       "mlstm",      # Standard mLSTM
       "moe-ffn",    # MoE FFN with 8 experts, top-2
       "mlstm",
       "moe-ffn"
   ]
   ```

2. **Multi-head mLSTM** (already have)
   - Our `num_heads` parameter already enables parallel heads
   - Each head has its own Q/K/V projections and state

3. **Hyena/Mamba Integration**
   ```python
   # Mixed architecture
   layer_types = [
       "mlstm",      # xLSTM layer
       "hyena",      # Efficient long-range
       "slstm",      # sLSTM variant
       "mamba"       # State-space model
   ]
   ```

4. **NCPS Mixer Integration**
   ```python
   # With NCPS wiring
   layer_types = [
       "mlstm",
       "ncps_cfc",   # NCPS Closed-form Continuous
       "mlstm",
       "ncps_ltc"    # Liquid Time Constant
   ]
   ```

### What to Implement

1. **BlockRegistry with MoE support**
   ```python
   class BlockRegistry:
       _block_types = {
           "mlstm": mLSTMBlock,
           "slstm": sLSTMBlock,
           "ffn": FFNBlock,
           "moe-ffn": MoEFFNBlock,      # ← New
           "hyena": HyenaBlock,          # ← New
           "mamba": MambaBlock,          # ← New
           "ncps_cfc": NCPSCfCBlock,     # ← New
       }
   ```

2. **Config-driven heterogeneous backbone** (like MAD + LFM2)
   ```python
   config = xLSTMConfig(
       num_blocks=32,
       layer_types=["mlstm", "moe-ffn"] * 16,
       moe_config={
           "num_experts": 8,
           "active_experts": 2
       }
   )
   ```

3. **Internal NCPS wiring** (within mLSTM blocks)
   - Wire Q/K/V projections with sparsity
   - Wire gates with polarity
   - Component-level flexibility

### What NOT to Implement

1. ❌ **Block-to-block parallelism** - Sequential dependencies make it impossible
2. ❌ **Threading/async for blocks** - Python GIL + sequential state updates
3. ❌ **Over-complex wiring** - Keep backbone simple like MAD/LFM2

---

## Recommended Implementation Order

Based on MAD's patterns and our needs:

### Phase 1: Core Architecture (Highest Priority)
1. **Fix dtype issue in mLSTM kernel** ← BLOCKING INFERENCE
2. **Create BlockRegistry** (mLSTM, sLSTM, FFN types)
3. **Implement config-driven backbone** (LFM2 `layer_types` pattern)
4. **Test with xLSTM-7B weights** (all mLSTM blocks)
5. **Test with xLSTM-1B weights** (mixed mLSTM/sLSTM blocks)

### Phase 2: Component Extraction
6. **Extract mLSTM components** (Q/K/V, gates, norms as separate modules)
7. **Add strategy registry for kernels** (NCPS callable pattern)
8. **Add HyperProfiles** (MLX↔PyTorch numerical equivalence)

### Phase 3: Extensions (Research/Future)
9. **Add MoE FFN block type** (MAD pattern)
10. **Add NCPS wiring within blocks** (sparsity masks, polarity)
11. **Add Hyena/Mamba block types** (MAD integration)
12. **Add NCPS mixer blocks** (CfC, LTC as block types)

---

## Conclusion

MAD uses **three forms of parallelism**, all happening **within layers**:

1. **MoE**: Multiple experts computed in parallel, top-k routing
2. **Multi-head**: Parallel attention/hyena heads
3. **Algorithm**: Choice of parallel vs sequential implementations

None of these involve **block-to-block parallelism** - that would require fundamentally different architectures (e.g., DAG-based, not sequential).

For xLSTM-MAD-NCPS, we should:
- ✅ Adopt MoE patterns for flexibility
- ✅ Keep multi-head parallelism (already have)
- ✅ Use config-driven heterogeneous blocks (LFM2 + MAD pattern)
- ✅ Add NCPS wiring **inside** blocks (not between)
- ❌ Don't try to parallelize block execution (sequential dependencies)

**Next step:** Fix the dtype issue, then implement the config-driven backbone with BlockRegistry.
