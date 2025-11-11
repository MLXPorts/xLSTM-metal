# xLSTM-7B Architecture Mapping for MAD

Complete specification for loading and running xLSTM-7B weights with MAD.

## Model Configuration

From `xlstm_7b_model/config.json`:

```json
{
  "embedding_dim": 4096,
  "num_heads": 8,
  "num_blocks": 32,
  "vocab_size": 50304,
  "head_dim": 512,           // v_dim / num_heads = 4096 / 8

  "qk_dim_factor": 0.5,      // qk_dim = 4096 * 0.5 = 2048
  "v_dim_factor": 1.0,       // v_dim = 4096 * 1.0 = 4096

  "ffn_proj_factor": 2.667,  // ffn_up = 4096 * 2.667 = 10923
  "ffn_round_up_to_multiple_of": 64,  // -> 10944

  "gate_soft_cap": 15.0,
  "output_logit_soft_cap": 30.0,

  "chunk_size": 64,
  "chunkwise_kernel": "chunkwise--triton_xl_chunk",
  "weight_mode": "single"
}
```

## Weight Structure

Each block has this exact structure (from `model.safetensors.index.json`):

```
backbone.blocks.{i}.norm_mlstm.weight                  [4096]
backbone.blocks.{i}.mlstm_layer.q.weight               [2048, 4096]
backbone.blocks.{i}.mlstm_layer.k.weight               [2048, 4096]
backbone.blocks.{i}.mlstm_layer.v.weight               [4096, 4096]
backbone.blocks.{i}.mlstm_layer.igate_preact.weight    [8, 4096]
backbone.blocks.{i}.mlstm_layer.igate_preact.bias      [8]
backbone.blocks.{i}.mlstm_layer.fgate_preact.weight    [8, 4096]
backbone.blocks.{i}.mlstm_layer.fgate_preact.bias      [8]
backbone.blocks.{i}.mlstm_layer.ogate_preact.weight    [4096, 4096]
backbone.blocks.{i}.mlstm_layer.multihead_norm.weight  [8, 512]
backbone.blocks.{i}.mlstm_layer.out_proj.weight        [4096, 4096]
backbone.blocks.{i}.norm_ffn.weight                    [4096]
backbone.blocks.{i}.ffn.proj_up.weight                 [10944, 4096]
backbone.blocks.{i}.ffn.proj_up_gate.weight            [10944, 4096]
backbone.blocks.{i}.ffn.proj_down.weight               [4096, 10944]

embedding.weight                                        [50304, 4096]
lm_head.weight                                          [50304, 4096]
post_blocks_norm.weight                                 [4096]  (if add_post_blocks_norm)
```

## Block Architecture

### mLSTMBlock Structure

```python
class mLSTMBlock:
    # Pre-normalization
    norm_mlstm: RMSNorm(4096, eps=1e-6)

    # mLSTM Layer
    mlstm_layer:
        # QKV projections
        q: Linear(4096 -> 2048, bias=False)
        k: Linear(4096 -> 2048, bias=False)
        v: Linear(4096 -> 4096, bias=False)

        # Gates (PER-HEAD!)
        igate_preact: Linear(4096 -> 8, bias=True)    # 8 heads
        fgate_preact: Linear(4096 -> 8, bias=True)    # 8 heads
        ogate_preact: Linear(4096 -> 4096, bias=False)

        # Multi-head normalization
        multihead_norm: MultiHeadLayerNorm(8 heads, 512 head_dim)

        # Output projection
        out_proj: Linear(4096 -> 4096, bias=False)

    # FFN pre-normalization
    norm_ffn: RMSNorm(4096, eps=1e-6)

    # SwiGLU FFN
    ffn:
        proj_up: Linear(4096 -> 10944, bias=False)
        proj_up_gate: Linear(4096 -> 10944, bias=False)
        proj_down: Linear(10944 -> 4096, bias=False)
```

### mLSTM Forward Pass

```python
def mlstm_layer_forward(x):
    # x: [B, S, 4096]

    # 1. QKV projections
    q = q_proj(x)           # [B, S, 2048]
    k = k_proj(x)           # [B, S, 2048]
    v = v_proj(x)           # [B, S, 4096]

    # 2. Gate projections
    i_preact = igate_preact(x)  # [B, S, 8]
    f_preact = fgate_preact(x)  # [B, S, 8]
    o_preact = ogate_preact(x)  # [B, S, 4096]

    # 3. Soft-cap gates (15.0)
    i_preact = soft_cap(i_preact, 15.0)
    f_preact = soft_cap(f_preact, 15.0)

    # 4. Reshape for multi-head
    q = q.reshape(B, S, 8, 256).transpose(1, 2)  # [B, 8, S, 256]
    k = k.reshape(B, S, 8, 256).transpose(1, 2)  # [B, 8, S, 256]
    v = v.reshape(B, S, 8, 512).transpose(1, 2)  # [B, 8, S, 512]
    i_preact = i_preact.transpose(1, 2)          # [B, 8, S]
    f_preact = f_preact.transpose(1, 2)          # [B, 8, S]

    # 5. mLSTM backend (TFLA kernel)
    h, state = mlstm_backend(
        q=q, k=k, v=v,
        i=i_preact, f=f_preact,
        c_initial=c_state,
        n_initial=n_state,
        m_initial=m_state
    )
    # h: [B, 8, S, 512]
    # state: (C_t, n_t, m_t)

    # 6. Transpose back and normalize
    h = h.transpose(1, 2)                    # [B, S, 8, 512]
    h_norm = multihead_norm(h)               # [B, S, 8, 512]
    h_norm = h_norm.reshape(B, S, 4096)      # [B, S, 4096]

    # 7. Output gate (sigmoid)
    h_out = sigmoid(o_preact) * h_norm       # [B, S, 4096]

    # 8. Output projection
    y = out_proj(h_out)                      # [B, S, 4096]

    return y, state
```

### FFN Forward Pass (SwiGLU)

```python
def ffn_forward(x):
    # x: [B, S, 4096]

    # SwiGLU: silu(gate) * up
    gate = proj_up_gate(x)    # [B, S, 10944]
    up = proj_up(x)           # [B, S, 10944]

    hidden = silu(gate) * up  # [B, S, 10944]

    y = proj_down(hidden)     # [B, S, 4096]

    return y
```

### Full Block Forward Pass

```python
def block_forward(x, state):
    # x: [B, S, 4096]

    # mLSTM path (with residual)
    x_mlstm = norm_mlstm(x)
    x_mlstm, state = mlstm_layer(x_mlstm, state)
    x = x + x_mlstm

    # FFN path (with residual)
    x_ffn = norm_ffn(x)
    x_ffn = ffn(x_ffn)
    x = x + x_ffn

    return x, state
```

## State Management

Each mLSTM layer maintains a 3-tuple state:

```python
state = (C_t, n_t, m_t)

C_t: Covariance matrix     [B, num_heads, head_dim, head_dim]  = [B, 8, 512, 512]
n_t: Normalizer            [B, num_heads, head_dim]             = [B, 8, 512]
m_t: Running max (log-space) [B, num_heads]                     = [B, 8]
```

Full model state:

```python
model_state = {
    0: (C_0, n_0, m_0),   # Block 0 state
    1: (C_1, n_1, m_1),   # Block 1 state
    ...
    31: (C_31, n_31, m_31)  # Block 31 state
}
```

## Key Implementation Details

### 1. Per-Head Gating (CRITICAL!)

**NOT** per-feature gating like standard LSTM:

```python
igate_preact: Linear(4096 -> 8)     # 8 outputs for 8 heads
fgate_preact: Linear(4096 -> 8)     # 8 outputs for 8 heads
```

Each head gets **one** scalar gate value, not per-dimension gates.

### 2. Soft Capping

```python
def soft_cap(x, cap_value):
    return cap_value * torch.tanh(x / cap_value)

# Gates: cap at 15.0
i_preact = soft_cap(i_preact, 15.0)
f_preact = soft_cap(f_preact, 15.0)

# Output logits: cap at 30.0
logits = soft_cap(logits, 30.0)
```

### 3. Multi-Head Normalization

Not standard LayerNorm! Normalizes **per-head**:

```python
class MultiHeadLayerNorm:
    def __init__(self, num_heads=8, head_dim=512):
        self.weight = Parameter(torch.ones(num_heads, head_dim))
        # Shape: [8, 512]

    def forward(self, x):
        # x: [B, S, num_heads, head_dim] = [B, S, 8, 512]
        # Normalize independently per head
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / sqrt(var + eps)
        return self.weight * x_norm
```

### 4. Dimension Calculations

```python
embedding_dim = 4096
num_heads = 8

# QK dimensions
qk_dim = int(embedding_dim * 0.5) = 2048
qk_head_dim = qk_dim // num_heads = 256

# V dimensions
v_dim = int(embedding_dim * 1.0) = 4096
v_head_dim = v_dim // num_heads = 512

# FFN dimensions
ffn_dim = int(embedding_dim * 2.667) = 10923
ffn_dim_rounded = round_up_to_64(10923) = 10944
```

### 5. Weight Mode: "single"

The 7B model uses `weight_mode: "single"`, meaning:

- Separate `q`, `k`, `v` weight matrices
- Separate `igate_preact`, `fgate_preact`, `ogate_preact` matrices

(NOT fused into single QKV projection)

## Inference Pipeline

```python
# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlstm_7b_model/")

# 2. Tokenize input
input_ids = tokenizer.encode(prompt)  # [1, S]

# 3. Embedding
x = embedding(input_ids)  # [1, S, 4096]

# 4. Process through 32 blocks
state = {}  # Empty initial state
for i, block in enumerate(blocks):
    x, block_state = block(x, state.get(i))
    state[i] = block_state

# 5. Final normalization (if enabled)
if add_post_blocks_norm:
    x = post_blocks_norm(x)

# 6. Language model head
logits = lm_head(x)  # [1, S, 50304]

# 7. Soft-cap output
logits = soft_cap(logits, 30.0)

# 8. Sample next token
next_token_logits = logits[0, -1, :]  # [50304]
next_token = sample(next_token_logits)

# 9. Continue generation with state
```

## Safetensors Loading

```python
from safetensors.torch import load_file

# Load all shards
weights = {}
for i in range(1, 7):
    shard = load_file(f"model-0000{i}-of-00006.safetensors")
    weights.update(shard)

# Map to model
for i in range(32):
    block.norm_mlstm.weight = weights[f"backbone.blocks.{i}.norm_mlstm.weight"]
    block.mlstm_layer.q.weight = weights[f"backbone.blocks.{i}.mlstm_layer.q.weight"]
    # ... etc
```

## Tokenizer

From `xlstm_7b_model/tokenizer.json`:

- Type: BPE (Byte-Pair Encoding)
- Vocab size: 50304
- Special tokens:
    - BOS: 0
    - EOS: 2
    - PAD: 1

## References

Source code locations for canonical implementation:

- `xlstm-solace-torch/src/xlstm_solace_torch/models/model.py` - Main model
- `xlstm-solace-torch/src/xlstm_solace_torch/models/mlstm/layer.py` - mLSTM layer
- `xlstm-solace-torch/src/xlstm_solace_torch/models/components.py` - MultiHeadLayerNorm, soft_cap
- `convert_hf_to_mlx.py` - Weight mapping reference
