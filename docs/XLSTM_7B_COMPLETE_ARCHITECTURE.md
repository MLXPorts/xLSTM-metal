# xLSTM-7B Complete Architecture Reference

**Source**: `transformers.models.xlstm.modeling_xlstm` (HuggingFace Transformers)

This document defines the complete xLSTM-7B architecture based on the canonical transformers implementation.

## High-Level Model Structure

```
xLSTMForCausalLM
├── backbone: xLSTMModel
│   ├── embeddings: Embedding(vocab_size=50304, embedding_dim=4096)
│   ├── blocks: ModuleList[32 × xLSTMBlock]
│   └── out_norm: RMSNorm(4096)
└── lm_head: Linear(4096 → 50304, bias=False)
    └── soft_cap(logits, cap_value=30.0)
```

## xLSTMBlock Architecture (CRITICAL)

Each xLSTMBlock contains **TWO** sub-blocks with residual connections:

```python
class xLSTMBlock:
    """
    Complete block structure (Lines 1166-1198 in modeling_xlstm.py)

    Each block processes:
    1. mLSTM path: norm → mlstm → residual
    2. FFN path: norm → ffn → residual
    """

    def __init__(self, config):
        # mLSTM components
        self.norm_mlstm = xLSTMRMSNorm(hidden_size=4096)
        self.mlstm_layer = xLSTMLayer(config)

        # FFN components (WE WERE MISSING THIS!)
        self.norm_ffn = xLSTMRMSNorm(hidden_size=4096)
        self.ffn = xLSTMFeedForward(config)

    def forward(self, x, state):
        # Path 1: mLSTM with residual
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm  # Residual connection

        # Path 2: FFN with residual (MISSING FROM OUR IMPLEMENTATION!)
        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn  # Residual connection

        return x, state
```

**Critical**: Each block has TWO residual streams, not one!

## xLSTMLayer (mLSTM Component)

Implements multi-head mLSTM with gates and normalization (Lines 1030-1164):

```python
class xLSTMLayer:
    """
    Core mLSTM layer - CORRECT in our current implementation!

    Key properties:
    - NO CausalConv1d (only in older/smaller xLSTM models)
    - NO up-projection split (only in older/smaller models)
    - Direct Q/K/V projections from input
    - Soft-capped gates
    """

    def __init__(self, config):
        # Dimensions
        self.v_dim = int(hidden_size * v_dim_factor)  # 5440 = 4096 * 1.328125
        self.qk_dim = int(hidden_size * qk_dim_factor)  # 2048 = 4096 * 0.5

        # Projections (weight_mode="single")
        self.q = Linear(4096 → 2048, bias=False)
        self.k = Linear(4096 → 2048, bias=False)
        self.v = Linear(4096 → 5440, bias=False)

        # Gates
        self.igate_preact = Linear(4096 → num_heads=8, bias=True)
        self.fgate_preact = Linear(4096 → num_heads=8, bias=True)
        self.ogate_preact = Linear(4096 → 5440, bias=False)

        # Backend kernel
        self.mlstm_backend = xLSTMBackend(config)

        # Multi-head normalization
        self.multihead_norm = xLSTMMultiHeadLayerNorm(
            num_heads=8,
            head_dim=5440//8=680
        )

        # Output projection
        self.out_proj = Linear(5440 → 4096, bias=False)

    def forward(self, x, state):
        # Project Q/K/V and gates
        query = q(x)  # [B, S, 2048]
        key = k(x)    # [B, S, 2048]
        value = v(x)  # [B, S, 5440]

        i_preact = soft_cap(igate_preact(x), cap_value=15.0)  # [B, S, 8]
        f_preact = soft_cap(fgate_preact(x), cap_value=15.0)  # [B, S, 8]
        o_preact = ogate_preact(x)  # [B, S, 5440]

        # Reshape to multi-head format
        # [B, S, D] → [B, NH, S, DH]
        query = query.view(B, S, NH, QK_DH).transpose(1, 2)
        key = key.view(B, S, NH, QK_DH).transpose(1, 2)
        value = value.view(B, S, NH, V_DH).transpose(1, 2)
        i_preact = i_preact.transpose(1, 2)  # [B, NH, S]
        f_preact = f_preact.transpose(1, 2)  # [B, NH, S]

        # mLSTM kernel (chunkwise/recurrent/step)
        h, state = mlstm_backend(
            query, key, value,
            igate=i_preact,
            fgate=f_preact,
            c_initial=state[0],
            n_initial=state[1],
            m_initial=state[2]
        )
        # h: [B, NH, S, V_DH]

        # Multi-head normalization
        h = h.transpose(1, 2)  # [B, S, NH, V_DH]
        h_norm = multihead_norm(h)  # [B, S, v_dim]

        # Output gating
        h_out = sigmoid(o_preact) * h_norm  # [B, S, 5440]

        # Output projection
        y = out_proj(h_out)  # [B, S, 4096]

        return y, state
```

## xLSTMFeedForward (FFN Component)

Gated FFN similar to Llama/GPT (Lines 983-1028):

```python
class xLSTMFeedForward:
    """
    Feed-forward network with gating.

    Architecture: x → SiLU(gate(x)) * up(x) → down(x)
    Similar to Llama's SwiGLU but with separate projections.
    """

    def __init__(self, config):
        # Up-projection dimension
        # For xLSTM-7B: 4096 * 2.6484375 = 10848, rounded to 10880
        self.up_proj_dim = round_up_to_next_multiple_of(
            config.hidden_size * config.ffn_proj_factor,  # 4096 * 2.6484375
            config.ffn_round_up_to_multiple_of  # 32
        )  # → 10880

        # Weight mode: "single" or "fused"
        if config.weight_mode == "single":
            # Separate projections (xLSTM-7B uses this)
            self.proj_up_gate = Linear(4096 → 10880, bias=False)
            self.proj_up = Linear(4096 → 10880, bias=False)
        elif config.weight_mode == "fused":
            # Fused projection (saves kernel launches)
            self.proj_up_gate_z = Linear(4096 → 2*10880, bias=False)

        self.proj_down = Linear(10880 → 4096, bias=False)
        self.act_fn = SiLU()

    def forward(self, x):
        if config.weight_mode == "single":
            # Separate gates: SiLU(gate) * value
            gate = self.proj_up_gate(x)  # [B, S, 10880]
            z = self.proj_up(x)          # [B, S, 10880]
            x = self.act_fn(gate) * z    # [B, S, 10880]

        elif config.weight_mode == "fused":
            # Fused gates (like Llama)
            x = self.proj_up_gate_z(x)   # [B, S, 2*10880]
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z    # [B, S, 10880]

        # Down projection
        y = self.proj_down(x)  # [B, S, 4096]
        return y
```

## Configuration Values (xLSTM-7B)

From `NX-AI/xLSTM-7b` HuggingFace model:

```python
config = xLSTMConfig(
    vocab_size=50304,
    hidden_size=4096,       # embedding_dim
    num_hidden_layers=32,   # num_blocks
    num_heads=8,

    # mLSTM dimensions
    v_dim_factor=1.328125,  # v_dim = 4096 * 1.328125 = 5440
    qk_dim_factor=0.5,      # qk_dim = 4096 * 0.5 = 2048

    # FFN dimensions
    ffn_proj_factor=2.6484375,  # up_proj_dim = 4096 * 2.6484375 ≈ 10848
    ffn_round_up_to_multiple_of=32,  # → rounds to 10880

    # Gates
    gate_soft_cap=15.0,
    output_logit_soft_cap=30.0,

    # Kernel settings
    chunk_size=64,
    eps=1e-6,
    mode="inference",
    autocast_kernel_dtype="bfloat16",
    inference_state_dtype="float32",

    # Normalization
    norm_eps=1e-6,
    norm_reduction_force_float32=True,

    # Other
    use_bias=False,
    weight_mode="single",
    return_last_states=True,
    max_inference_chunksize=2048
)
```

## Weight Keys in Safetensors

For each block `i` in `[0, 31]`:

```
# Embedding
backbone.embeddings.weight: [50304, 4096]

# Block i components
backbone.blocks.{i}.norm_mlstm.weight: [4096]
backbone.blocks.{i}.norm_ffn.weight: [4096]

# mLSTM layer
backbone.blocks.{i}.mlstm_layer.q.weight: [2048, 4096]
backbone.blocks.{i}.mlstm_layer.k.weight: [2048, 4096]
backbone.blocks.{i}.mlstm_layer.v.weight: [5440, 4096]
backbone.blocks.{i}.mlstm_layer.igate_preact.weight: [8, 4096]
backbone.blocks.{i}.mlstm_layer.igate_preact.bias: [8]
backbone.blocks.{i}.mlstm_layer.fgate_preact.weight: [8, 4096]
backbone.blocks.{i}.mlstm_layer.fgate_preact.bias: [8]
backbone.blocks.{i}.mlstm_layer.ogate_preact.weight: [5440, 4096]
backbone.blocks.{i}.mlstm_layer.multihead_norm.weight: [5440]
backbone.blocks.{i}.mlstm_layer.out_proj.weight: [4096, 5440]

# FFN
backbone.blocks.{i}.ffn.proj_up_gate.weight: [10880, 4096]
backbone.blocks.{i}.ffn.proj_up.weight: [10880, 4096]
backbone.blocks.{i}.ffn.proj_down.weight: [4096, 10880]

# Output
backbone.out_norm.weight: [4096]
lm_head.weight: [50304, 4096]
```

## Key Differences from Older xLSTM Models

xLSTM-7B (transformers) vs older xLSTM (xlstm package):

| Component | xLSTM-7B (NEW) | Older xLSTM (xlstm package) |
|-----------|----------------|----------------------------|
| Conv1d | ❌ NOT used | ✅ CausalConv1d before mLSTM |
| Up-projection | ❌ Direct projections | ✅ Up-project + split |
| Skip connections | ❌ None | ✅ Learnable skip in mLSTM |
| FFN | ✅ Separate FFN block | ❌ Or combined with mLSTM |
| QK scaling | ✅ Scale Q during retrieval | ⚠️ Varies (often scale K) |

**CRITICAL**: Do NOT mix architectures! xLSTM-7B uses the transformers approach.

## MLX Implementation TODO

Our current MLX implementation is **MISSING**:

1. ✅ **xLSTMLayer** - CORRECT! (Direct Q/K/V, no conv1d)
2. ❌ **xLSTMFeedForward** - MISSING! Need gated FFN
3. ❌ **xLSTMBlock** - MISSING! Need to wire: norm_mlstm → mlstm → residual → norm_ffn → ffn → residual

### Implementation Plan

1. Create `mad/blocks/ffn_mlx/block.py` with xLSTMFeedForward
2. Update `mad/models/xlstm_7b_mlx.py` to include FFN in each block
3. Verify weight loading matches safetensors keys
4. Test output quality improves

## References

- Transformers source: `/opt/homebrew/anaconda3/lib/python3.13/site-packages/transformers/models/xlstm/modeling_xlstm.py`
- HuggingFace model: `NX-AI/xLSTM-7b`
- Config: Lines 1-50 in modeling_xlstm.py
- xLSTMBlock: Lines 1166-1198
- xLSTMLayer: Lines 1030-1164
- xLSTMFeedForward: Lines 983-1028
