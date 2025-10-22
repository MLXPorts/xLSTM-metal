# MAD Composition Extension Proposal

**Aligning with LFM2's Layer Types Pattern**

Date: 2025-01-21
Status: Proposal
Author: Claude (based on Sepp Hochreiter's MAD and LFM2 architectures)

## Executive Summary

LFM2 (Liquid Foundation Model 2) and MAD (Mechanistic Architecture Design) were both created by Sepp Hochreiter's research group. LFM2 uses a simple but powerful `layer_types` pattern for heterogeneous block composition that we should adopt and extend for MAD.

## Current State

### MAD's Current Composition (Sequential Only)

```python
# mad/registry.py
layer_registry = {
    'mlstm': {'module': MLSTMBlock, ...},
    'chunkwise-mlstm': {'module': ChunkwiseMLSTM, ...},
    'swiglu': {'module': SwiGLU, ...},
    'attention': {'module': Attention, ...},
}

# Config (implicit sequential)
layers = ['mlstm', 'swiglu', 'chunkwise-mlstm', 'swiglu']
```

**Limitations:**
- Sequential only (no parallelism, no branching)
- No explicit weight sharing
- No fork/join patterns
- Backend selection is global (not per-layer)

## LFM2's Approach (Reference Implementation)

### 1. Configuration Pattern

```python
# transformers/models/lfm2/configuration_lfm2.py
class Lfm2Config(PretrainedConfig):
    def __init__(
        self,
        num_hidden_layers: int = 32,
        layer_types: Optional[list[str]] = None,
        full_attn_idxs: Optional[list[int]] = None,
        **kwargs,
    ):
        self.layer_types = layer_types
        if self.layer_types is None:
            # Default: full_attention at specified indices, conv elsewhere
            full_attn_idxs = full_attn_idxs or list(range(num_hidden_layers))
            self.layer_types = [
                "full_attention" if i in full_attn_idxs else "conv"
                for i in range(num_hidden_layers)
            ]
```

**Example config:**
```python
layer_types = [
    "full_attention",  # Layer 0
    "conv",            # Layer 1
    "conv",            # Layer 2
    "full_attention",  # Layer 3
    "conv",            # Layer 4
    # ... 32 layers total
]
```

### 2. Per-Layer Polymorphism

```python
# transformers/models/lfm2/modular_lfm2.py
class Lfm2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()
        # Conditional construction based on layer_idx
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ConvLayer(config, layer_idx)

        self.input_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = Lfm2MLP(config)
        self.post_attn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
```

### 3. Hybrid Cache Management

```python
class Lfm2HybridConvCache:
    """Manages both attention KV cache and conv state cache"""
    def __init__(self, config, max_batch_size, dtype, device):
        self.layer_types = config.layer_types
        self.first_attention_layer = self.layer_types.index("full_attention")

        # Key-value cache for attention layers
        self.key_cache = []
        self.value_cache = []

        # Conv state cache for conv layers
        self.conv_states = {}
```

### 4. Model Construction

```python
class Lfm2Model(LlamaModel):
    def __init__(self, config: Lfm2Config):
        super().__init__(config)  # Builds layers via parent
        # Parent calls: self.layers = nn.ModuleList([...])
        # Transformers' modular system resolves LlamaDecoderLayer -> Lfm2DecoderLayer
```

**Key insight:** The parent `LlamaModel.__init__` calls:
```python
self.layers = nn.ModuleList(
    [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
)
```

But Transformers' modular resolution automatically substitutes `Lfm2DecoderLayer` based on the model type.

## Proposed MAD Extension

### Phase 1: Adopt LFM2's layer_types Pattern (Simple)

**Goal:** Enable heterogeneous sequences with per-layer backend selection.

```python
# mad/config/architecture.yml
architecture:
  num_layers: 12
  layer_types:
    - type: "mlstm"
      backend: "mlx"
      params:
        num_heads: 8
        head_dim: 64
    - type: "swiglu"
      backend: "mlx"
    - type: "chunkwise-mlstm"
      backend: "pytorch"
      params:
        chunk_size: 64
    - type: "swiglu"
      backend: "mlx"
    # Repeat pattern...
```

**Implementation:**

```python
# mad/model/blocks.py
class MADBlock(nn.Module):
    """Unified block that conditionally constructs based on layer_type"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        layer_spec = config.layer_types[layer_idx]
        layer_type = layer_spec['type']
        backend = layer_spec.get('backend', 'mlx')  # Default to MLX

        # Get module from registry
        registry_entry = layer_registry[layer_type]
        if 'backends' in registry_entry:
            module_class = registry_entry['backends'][backend]
        else:
            module_class = registry_entry['module']

        # Instantiate with layer-specific params
        self.layer = module_class(config, **layer_spec.get('params', {}))
        self.layer_type = layer_type
        self.backend = backend

    def forward(self, x, hidden_state=None):
        return self.layer(x, hidden_state)
```

```python
# mad/model/language_model.py
class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = Embedding(config.vocab_size, config.d_model)

        # Build heterogeneous layer stack
        self.layers = nn.ModuleList([
            MADBlock(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
```

**Benefits:**
- ✅ Per-layer backend selection (MLX, PyTorch, torch.compile)
- ✅ Per-layer hyperparameters
- ✅ Heterogeneous sequences (mlstm, attention, conv, etc.)
- ✅ Minimal code changes
- ✅ Aligns with LFM2's proven pattern

**Limitations:**
- Still sequential only
- No weight sharing
- No parallelism

### Phase 2: Extend to "Bricks" for Computation Graphs (Advanced)

**Goal:** Support fork/join, parallelism, and weight sharing.

```python
# mad/config/architecture.yml
architecture:
  bricks:
    # Primitive building blocks (weights + ops)
    mlstm_1:
      type: "mlstm"
      backend: "mlx"
      params: {num_heads: 8, head_dim: 64}

    mlstm_2:
      type: "mlstm"
      backend: "mlx"
      params: {num_heads: 8, head_dim: 64}

    swiglu:
      type: "swiglu"
      backend: "mlx"

    conv:
      type: "conv"
      backend: "pytorch"

  blocks:
    # Composition patterns (how bricks connect)
    - type: "sequential"
      bricks: ["mlstm_1", "swiglu"]

    - type: "parallel"
      bricks: ["mlstm_2", "conv"]
      combiner: "add"  # or "concat", "gate"

    - type: "sequential"
      bricks: ["swiglu"]
```

**Visualization:**
```
Input
  ↓
[mlstm_1] → [swiglu]
  ↓
[mlstm_2]
  ↓        ↘
         (parallel, add)
  ↓        ↗
[conv]
  ↓
[swiglu]
  ↓
Output
```

**Implementation sketch:**

```python
# mad/model/bricks.py
class Brick(nn.Module):
    """Primitive building block (weights + ops)"""
    def __init__(self, brick_config):
        super().__init__()
        self.layer = instantiate_from_registry(brick_config)

    def forward(self, x, hidden_state=None):
        return self.layer(x, hidden_state)

class Block(nn.Module):
    """Composition pattern (wiring of bricks)"""
    def __init__(self, block_config, bricks_dict):
        super().__init__()
        self.block_type = block_config['type']
        self.bricks = [bricks_dict[name] for name in block_config['bricks']]

        if self.block_type == 'parallel':
            self.combiner = block_config.get('combiner', 'add')

    def forward(self, x, hidden_state=None):
        if self.block_type == 'sequential':
            for brick in self.bricks:
                x, hidden_state = brick(x, hidden_state)
            return x, hidden_state

        elif self.block_type == 'parallel':
            outputs = [brick(x, hidden_state) for brick in self.bricks]

            if self.combiner == 'add':
                x = sum(out[0] for out in outputs)
                # Merge hidden states (implementation-dependent)
            elif self.combiner == 'concat':
                x = torch.cat([out[0] for out in outputs], dim=-1)

            return x, hidden_state
```

**Benefits:**
- ✅ Fork/join patterns
- ✅ Parallelism (multiple paths)
- ✅ Weight sharing (reference same brick multiple times)
- ✅ Declarative composition
- ✅ Graph-based architecture design

**Challenges:**
- More complex than layer_types
- Cache management for parallel paths
- Hidden state merging semantics
- Backend interop (MLX ↔ PyTorch)

## Recommendation

**Start with Phase 1** (layer_types pattern):

1. It's proven (LFM2 uses it in production)
2. Minimal code changes
3. Solves 80% of composition needs
4. Natural extension of MAD's current design
5. Same author group (Hochreiter)

**Then consider Phase 2** if needed:
- Only if evaluation shows parallelism is critical
- Only if weight sharing patterns emerge
- After Phase 1 is validated numerically

## Implementation Plan

### Step 1: Extend MAD Config
```python
# mad/config.py
class MADConfig:
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 12,
        layer_types: Optional[List[Dict]] = None,
        **kwargs
    ):
        self.layer_types = layer_types
        if self.layer_types is None:
            # Default: all mlstm
            self.layer_types = [
                {'type': 'mlstm', 'backend': 'mlx'}
                for _ in range(num_layers)
            ]
```

### Step 2: Create MADBlock
```python
# mad/model/blocks.py
class MADBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        # Implementation as shown above
        pass
```

### Step 3: Update LanguageModel
```python
# mad/model/language_model.py
class LanguageModel(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([
            MADBlock(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
```

### Step 4: Test Sequences

**Test 1: Homogeneous (baseline)**
```python
layer_types = [{'type': 'mlstm', 'backend': 'mlx'}] * 12
```

**Test 2: Alternating (like LFM2)**
```python
layer_types = [
    {'type': 'mlstm', 'backend': 'mlx'},
    {'type': 'swiglu', 'backend': 'mlx'},
] * 6  # 12 layers total
```

**Test 3: Hybrid backend**
```python
layer_types = [
    {'type': 'mlstm', 'backend': 'mlx'},
    {'type': 'chunkwise-mlstm', 'backend': 'pytorch'},
    {'type': 'swiglu', 'backend': 'mlx'},
] * 4  # 12 layers total
```

**Test 4: xLSTM[7:1] pattern (canonical)**
```python
layer_types = [
    {'type': 'mlstm', 'backend': 'mlx'},  # mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},  # FFN
    {'type': 'mlstm', 'backend': 'mlx'},  # mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},  # FFN
    {'type': 'mlstm', 'backend': 'mlx'},  # mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},  # FFN
    {'type': 'mlstm', 'backend': 'mlx'},  # mLSTM
    {'type': 'slstm', 'backend': 'mlx'},  # sLSTM
]
```

## Key Differences from LFM2

| Aspect | LFM2 | MAD (Proposed) |
|--------|------|----------------|
| Layer types | 2 (attention, conv) | N (mlstm, swiglu, attention, hyena, ...) |
| Backend | Single (PyTorch) | Multi (MLX, PyTorch, torch.compile) |
| Per-layer params | Limited | Full (num_heads, chunk_size, etc.) |
| State management | Hybrid cache | Per-layer hidden state |
| Primary use | Foundation model | Architecture evaluation |
| Extension | No (production) | Yes (research) |

## Conclusion

LFM2's `layer_types` pattern is the right foundation for MAD composition extension:

1. **Proven design** from the same research group
2. **Simple to implement** (~100 lines of code)
3. **Backward compatible** with current MAD
4. **Extensible** to Phase 2 (bricks) if needed

Next steps:
1. Implement Phase 1 (layer_types)
2. Define xLSTM canonical sequences (7:1, 1:1 patterns)
3. Test numerical parity against xlstm-large
4. Validate on MAD's 6 synthetic tasks

---

**References:**
- LFM2: `/Users/sydneybach/miniconda3/lib/python3.12/site-packages/transformers/models/lfm2/`
- MAD paper: Appendix (iso-state normalization, architecture zoo)
- xlstm-large: [canonical checkpoint for validation]
