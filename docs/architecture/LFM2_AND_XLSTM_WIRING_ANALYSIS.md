# LFM2 and xLSTM Wiring Analysis

## Executive Summary

After analyzing the LFM2 (Liquid Foundation Model 2) implementation in transformers and comparing it to MAD and xLSTM-7B, here are the key findings:

1. **No true parallelism** - All models use sequential block execution with residual connections
2. **Heterogeneous layers** - LFM2 uses a config-driven `layer_types` pattern for mixing different operator types
3. **Two-level architecture** - Blocks contain operators (attention/conv) + FFN with pre-norm residuals
4. **The "parallelism" is GPU-level** - Tiled computation within kernels, not block-to-block parallelism

## LFM2 Architecture Pattern

### Configuration-Driven Heterogeneity

```python
# configuration_lfm2.py lines 150-153
self.layer_types = layer_types
if self.layer_types is None:
    full_attn_idxs = full_attn_idxs if full_attn_idxs is not None else list(range(num_hidden_layers))
    self.layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(num_hidden_layers)]
```

**Key insight:** `layer_types` is a list like:
```python
["conv", "conv", "full_attention", "conv", "conv", "full_attention", ...]
```

Each layer can be either:
- `"full_attention"` - Multi-head attention with RoPE
- `"conv"` - Short depthwise convolution with gating

### Block Structure

```python
# modeling_lfm2.py lines 528-565
class Lfm2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        # Conditional operator construction
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)

        # Always have FFN
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states

        # Operator path (attention OR conv)
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                ...
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                ...
            )
        hidden_states = hidden_states + residual  # Residual connection

        # FFN path (always present)
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

        return hidden_states
```

### Model Composition

```python
# modeling_lfm2.py lines 606-608
self.layers = nn.ModuleList(
    [Lfm2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
)

# Forward pass (sequential iteration)
for layer in self.layers:
    hidden_states = layer(hidden_states, ...)
```

**No parallelism!** Just like MAD, LFM2 uses:
- `nn.ModuleList` to store layers
- Sequential iteration in forward pass
- Manual residual connections

## xLSTM-7B Structure (from weights)

Looking at `xlstm_7b_model/model.safetensors.index.json`:

```
backbone.blocks.0.norm_mlstm.weight
backbone.blocks.0.mlstm_layer.q.weight
backbone.blocks.0.mlstm_layer.k.weight
backbone.blocks.0.mlstm_layer.v.weight
backbone.blocks.0.mlstm_layer.igate_preact.weight
backbone.blocks.0.mlstm_layer.igate_preact.bias
backbone.blocks.0.mlstm_layer.fgate_preact.weight
backbone.blocks.0.mlstm_layer.fgate_preact.bias
backbone.blocks.0.mlstm_layer.ogate_preact.weight
backbone.blocks.0.mlstm_layer.multihead_norm.weight
backbone.blocks.0.mlstm_layer.out_proj.weight
backbone.blocks.0.norm_ffn.weight
backbone.blocks.0.ffn.proj_up.weight
backbone.blocks.0.ffn.proj_up_gate.weight
backbone.blocks.0.ffn.proj_down.weight
```

### Key Observations

1. **All 32 blocks are identical** - Every block has `norm_mlstm`, `mlstm_layer`, `norm_ffn`, `ffn`
2. **No sLSTM in inference** - Only mLSTM layers present
3. **Same pattern as LFM2** - Pre-norm → operator → pre-norm → FFN

### xLSTM-1B vs xLSTM-7B

The user mentioned xLSTM-1B has different block structure. Let's check:

```bash
# Would need to download xLSTM-1B to see if it has:
# - Different layer_types (mLSTM vs sLSTM mix?)
# - Different block depths
# - Different component configurations
```

## What "Parallelism" Actually Means

### GPU-Level Parallelism (What We Have)

All three systems (MAD, LFM2, xLSTM) use **GPU-level tiled computation**:

1. **Within attention** - Multiple heads computed in parallel
2. **Within convolution** - Depthwise groups computed in parallel
3. **Within mLSTM chunkwise** - Chunks processed in parallel

Example from our `fw_kernel_parallel.metal`:
```metal
// Each threadgroup processes a tile
// Multiple threadgroups run in parallel on GPU
kernel void mlstm_chunkwise_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    ...
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    // Parallel tile computation
}
```

### Block-to-Block Parallelism (What We DON'T Have)

Neither MAD, LFM2, nor canonical xLSTM have:
- Parallel execution of multiple blocks
- Threading/async for block processing
- Model parallelism (that's a separate concern - data/tensor parallelism)

**Why?** Sequential dependencies:
- Block N+1 depends on output of Block N
- State updates are sequential (especially for mLSTM recurrent state)

## Design Patterns for xLSTM-MAD-NCPS

### 1. Config-Driven Heterogeneous Blocks (from LFM2)

```python
class xLSTMConfig:
    """
    Flexible configuration supporting multiple model variants

    Examples:
        # xLSTM-7B: All mLSTM blocks
        layer_types = ["mlstm"] * 32

        # xLSTM-1B: Mixed mLSTM + sLSTM
        layer_types = ["mlstm", "slstm", "mlstm", "slstm", ...]

        # Future: Integrate NCPS mixers
        layer_types = ["mlstm", "ncps_cfc", "mlstm", "ffn_only", ...]
    """
    num_blocks: int = 32
    layer_types: List[str] = None  # If None, defaults to all mLSTM

    # Per-type configurations
    mlstm_config: dict = {...}
    slstm_config: dict = {...}
    ffn_config: dict = {...}
```

### 2. Block Registry (Combining MAD + LFM2 + NCPS Patterns)

```python
# xlstm_metal/blocks/registry.py
class BlockRegistry:
    """
    Registry for block types (similar to MAD's layer_registry)

    Supports:
    - mLSTM blocks
    - sLSTM blocks
    - FFN blocks
    - NCPS wired blocks (CfC, LTC, etc.)
    - Custom blocks
    """

    _block_types = {
        "mlstm": (mLSTMBlock, "mlstm_config"),
        "slstm": (sLSTMBlock, "slstm_config"),
        "ffn_only": (FFNOnlyBlock, "ffn_config"),
        "ncps_cfc": (NCPSCfCBlock, "ncps_config"),
        # ... extensible
    }

    @classmethod
    def build_block(cls, block_type: str, config, layer_idx: int):
        """Build block from type string"""
        if block_type not in cls._block_types:
            raise ValueError(f"Unknown block type: {block_type}")

        block_class, config_key = cls._block_types[block_type]
        block_config = getattr(config, config_key)
        return block_class(block_config, layer_idx)
```

### 3. Backbone Composition

```python
# xlstm_metal/backbone.py
class xLSTMBackbone(nn.Module):
    """
    Config-driven heterogeneous backbone (LFM2 pattern)
    """

    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        # Default to all mLSTM if not specified
        if config.layer_types is None:
            config.layer_types = ["mlstm"] * config.num_blocks

        # Build heterogeneous blocks from registry
        self.blocks = nn.ModuleList([
            BlockRegistry.build_block(
                block_type=config.layer_types[i],
                config=config,
                layer_idx=i
            )
            for i in range(config.num_blocks)
        ])

        # Embeddings and norms
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.out_norm = RMSNorm(config.embedding_dim, eps=config.norm_eps)

    def __call__(self, input_ids, states=None):
        """Sequential forward pass (MAD/LFM2 pattern)"""
        x = self.embedding(input_ids)

        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states else None
            x_block, new_state = block(x, state)
            x = x + x_block  # Residual connection
            new_states.append(new_state)

        x = self.out_norm(x)
        return x, new_states
```

### 4. Internal Block Wiring (NCPS Pattern)

**This is where NCPS comes in** - not for block-to-block, but for **internal component wiring**:

```python
# xlstm_metal/blocks/mlx/mlstm/block.py
class mLSTMBlock(nn.Module):
    """
    Single mLSTM block with NCPS-style internal wiring

    Components:
    - Q/K/V projections (can be wired with sparsity)
    - Input/Forget/Output gates (can be wired with polarity)
    - Memory update
    - Norms
    - FFN
    """

    def __init__(self, config, layer_idx):
        super().__init__()

        # Operator path (mLSTM layer with internal NCPS wiring)
        self.norm_mlstm = RMSNorm(config.embedding_dim, eps=config.norm_eps)
        self.mlstm_layer = mLSTMLayer(config)  # ← Has internal wiring

        # FFN path
        self.norm_ffn = RMSNorm(config.embedding_dim, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def __call__(self, x, state=None):
        """LFM2 pattern: operator + FFN with residuals"""
        # mLSTM path
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, new_state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm  # Residual

        # FFN path
        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn  # Residual

        return x, new_state
```

## Key Takeaways

### What We Should Do

1. **Use LFM2's `layer_types` pattern** for heterogeneous blocks
   - Supports xLSTM-1B (mixed mLSTM/sLSTM)
   - Supports xLSTM-7B (all mLSTM)
   - Future-proof for NCPS mixers

2. **Use MAD's registry pattern** for extensibility
   - Easy to add new block types
   - Config-driven construction
   - Clean separation of concerns

3. **Use NCPS wiring INSIDE blocks** (not between blocks)
   - Wire Q/K/V projections with sparsity masks
   - Wire gates with polarity
   - Enable component-level flexibility

4. **Use HyperProfiles** for backend compensation
   - MLX vs PyTorch numerical differences
   - Initialization ranges
   - Dtype handling

### What We Should NOT Do

1. **Don't try to parallelize blocks** - Sequential dependencies make it impossible
2. **Don't over-engineer the backbone** - Keep it simple like MAD/LFM2
3. **Don't break the config.json format** - Must load xLSTM-7B/1B weights correctly

## Proposed Implementation Order

1. **Fix dtype issue in mLSTM kernel** (HIGH PRIORITY - blocking inference)
2. **Create BlockRegistry** with mLSTM/sLSTM/FFN types
3. **Implement config-driven xLSTMBackbone** (LFM2 pattern)
4. **Extract mLSTM components** (Q/K/V, gates, norms)
5. **Add NCPS-style internal wiring** (optional, for research)
6. **Add HyperProfiles** for MLX↔PyTorch equivalence
7. **Add NCPS mixers as block types** (future work)

## Next Steps

Should we:
1. **Start with dtype fix** (unblock inference testing)
2. **Design the config system** (support xLSTM-1B/7B/future models)
3. **Implement BlockRegistry** (enable heterogeneous blocks)

Which priority?
