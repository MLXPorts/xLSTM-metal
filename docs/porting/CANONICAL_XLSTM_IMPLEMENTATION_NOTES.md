# Canonical xLSTM Implementation: Key Observations

**From the official xLSTM notebook showing real usage**

Date: 2025-01-21
Source: Official xLSTM repository example notebook

## Available Kernels (Critical!)

```python
get_available_mlstm_kernels()
# Returns:
['chunkwise--native_autograd',
 'chunkwise--native_custbw',
 'chunkwise--triton_limit_chunk',
 'chunkwise--triton_xl_chunk',        # ← TFLA kernels!
 'chunkwise--triton_xl_chunk_siging',
 'parallel--native_autograd',
 'parallel--native_custbw',
 'parallel--native_stablef_autograd',
 'parallel--native_stablef_custbw',
 'parallel--triton_limit_headdim',
 'parallel--native_siging_autograd',
 'parallel--native_siging_custbw']
```

### Key Observations

**1. Multiple Backend Options:**

- `chunkwise--*`: Sequential between chunks, parallel within chunks
- `parallel--*`: Fully parallel formulation (via parallel scan!)
- `native_*`: Pure PyTorch implementation
- `triton_*`: Triton CUDA kernels (highly optimized)

**2. The "xl_chunk" Variant:**

```python
chunkwise_kernel="chunkwise--triton_xl_chunk"  # xl_chunk == TFLA kernels
```

This is the **Tiled Flash Linear Attention** implementation - the most optimized chunkwise kernel.

**3. Three Kernel Types:**

- **Training kernel** (`chunkwise_kernel`): For forward/backward passes
- **Sequence kernel** (`sequence_kernel`): For processing full sequences
- **Step kernel** (`step_kernel`): For autoregressive generation (one token at a time)

## Configuration Pattern

```python
xlstm_config = xLSTMLargeConfig(
    embedding_dim=512,
    num_heads=4,
    num_blocks=6,
    vocab_size=2048,
    return_last_states=True,
    mode="inference",
    chunkwise_kernel="chunkwise--triton_xl_chunk",  # TFLA for training
    sequence_kernel="native_sequence__triton",      # Triton for sequences
    step_kernel="triton",                           # Triton for generation
)
```

**Key parameters:**

- `mode="inference"`: Optimized for generation (vs "train")
- `return_last_states=True`: Return hidden states for multi-step generation
- Different kernels for different use cases (flexibility!)

## Model Architecture Inspection

```python
xLSTMLarge(
  (embedding): Embedding(2048, 512)
  (backbone): xLSTMLargeBlockStack(
    (blocks): ModuleList(
      (0-5): 6 x mLSTMBlock(
        (norm_mlstm): RMSNorm()
        (mlstm_layer): mLSTMLayer(
          (q): Linear(in_features=512, out_features=256, bias=False)
          (k): Linear(in_features=512, out_features=256, bias=False)
          (v): Linear(in_features=512, out_features=512, bias=False)
          (ogate_preact): Linear(in_features=512, out_features=512, bias=False)
          (igate_preact): Linear(in_features=512, out_features=4, bias=True)
          (fgate_preact): Linear(in_features=512, out_features=4, bias=True)
          (ogate_act_fn): Sigmoid()
          (mlstm_backend): mLSTMBackend(...)
          (multihead_norm): MultiHeadLayerNorm()
          (out_proj): Linear(in_features=512, out_features=512, bias=False)
        )
        (norm_ffn): RMSNorm()
        (ffn): FeedForward(
          (proj_up_gate): Linear(in_features=512, out_features=1408, bias=False)
          (proj_up): Linear(in_features=512, out_features=1408, bias=False)
          (proj_down): Linear(in_features=1408, out_features=512, bias=False)
          (act_fn): SiLU()
        )
      )
    )
    (out_norm): RMSNorm()
  )
  (lm_head): Linear(in_features=512, out_features=2048, bias=False)
)
```

### Critical Observations

**1. Gate Dimensions:**

```python
(igate_preact): Linear(in_features=512, out_features=4, bias=True)
(fgate_preact): Linear(in_features=512, out_features=4, bias=True)
```

**Gates are per-head!** With 4 heads:

- Input gate: 4 outputs (1 per head)
- Forget gate: 4 outputs (1 per head)
- This enables **independent exponential scaling per head**

**2. Output Gate is Different:**

```python
(ogate_preact): Linear(in_features=512, out_features=512, bias=False)
```

Output gate is **per-feature**, not per-head. This is sigmoid-activated, not exponential.

**3. Pre-Normalization (Megatron-style):**

```python
(norm_mlstm): RMSNorm()  # BEFORE mlstm_layer
(mlstm_layer): mLSTMLayer(...)
# No norm after - just residual connection
```

This confirms the Megatron BERT fix we discussed.

**4. FFN Structure (SwiGLU variant):**

```python
(proj_up_gate): Linear(...)  # Gate projection
(proj_up): Linear(...)       # Value projection
(proj_down): Linear(...)     # Down projection
(act_fn): SiLU()
```

This is **gated GLU**: `output = down_proj(silu(up_gate) * up_proj)`

## Dual Mode Operation

### Mode 1: Chunkwise (Parallel Training)

```python
# Process entire sequence at once
input = torch.randint(0, 2048, (3, 256))  # [batch=3, seq=256]
out_chunkwise, last_state = xlstm(input)
# out_chunkwise.shape = [3, 256, 2048]  # Full sequence output
```

**What happens internally:**

- Split sequence into chunks of 64 tokens (default `chunk_size=64`)
- Process chunks: 256 / 64 = 4 chunks
- **Parallel within each chunk** (uses TFLA/Triton kernels)
- **Sequential between chunks** (recurrent state dependency)

### Mode 2: Step-by-Step (Autoregressive Generation)

```python
# Process one token at a time
state = None
for i in range(256):
    out_step, state = xlstm(input[:, i:i+1], state)
    # out_step.shape = [3, 1, 2048]  # Single token output
```

**What happens internally:**

- Uses `step_kernel` (optimized for single-step)
- Maintains state across calls
- No chunking needed (already single token)
- Efficient for generation

## Numerical Parity Test

```python
# Chunkwise (parallel)
out_chunkwise, _ = xlstm(input)

# Step-by-step (sequential)
out_steps = []
state = None
for i in range(256):
    out_step, state = xlstm(input[:, i:i+1], state)
    out_steps.append(out_step)
out_steps = torch.cat(out_steps, dim=1)

# Compare
max_diff = (out_chunkwise - out_steps).abs().max()
# Result: 0.0085 (excellent!)

torch.allclose(out_chunkwise, out_steps, atol=7e-2, rtol=1e-3)
# Result: True
```

**Key insight:** Chunkwise and step-by-step are **numerically equivalent** (within tolerances).

This validates that:

1. Chunking is correct (no approximation)
2. State management is correct
3. Both paths use same underlying computation

## State Structure

```python
state.keys()
# dict_keys([0, 1, 2, 3, 4, 5])  # One per block

len(state), len(state[0])
# (6, 3)  # 6 blocks, 3 state tensors each
```

**Each block's state is a tuple of 3 tensors:**

1. **C_t**: Covariance matrix (memory state)
2. **n_t**: Normalizer vector
3. **m_t**: Running max for stabilization (the exponential scaling factor!)

This matches our canonical implementation exactly.

## Weight Structure

```python
list(xlstm.state_dict().keys())
# For each block (6 total):
'backbone.blocks.0.norm_mlstm.weight'              # Pre-norm (RMSNorm)
'backbone.blocks.0.mlstm_layer.q.weight'           # Query projection
'backbone.blocks.0.mlstm_layer.k.weight'           # Key projection
'backbone.blocks.0.mlstm_layer.v.weight'           # Value projection
'backbone.blocks.0.mlstm_layer.ogate_preact.weight'  # Output gate
'backbone.blocks.0.mlstm_layer.igate_preact.weight'  # Input gate (per-head)
'backbone.blocks.0.mlstm_layer.igate_preact.bias'    # Input gate bias
'backbone.blocks.0.mlstm_layer.fgate_preact.weight'  # Forget gate (per-head)
'backbone.blocks.0.mlstm_layer.fgate_preact.bias'    # Forget gate bias
'backbone.blocks.0.mlstm_layer.multihead_norm.weight' # Per-head norm
'backbone.blocks.0.mlstm_layer.out_proj.weight'      # Output projection
'backbone.blocks.0.norm_ffn.weight'                # FFN pre-norm
'backbone.blocks.0.ffn.proj_up_gate.weight'        # FFN gate
'backbone.blocks.0.ffn.proj_up.weight'             # FFN up
'backbone.blocks.0.ffn.proj_down.weight'           # FFN down
```

**Weight sharing pattern:**

- `embedding.weight` (input embedding)
- `lm_head.weight` (output embedding)
- These are **separate** (not tied) in this implementation

## Implications for Our MLX Implementation

### 1. We Need Multiple Kernels

**For MAD/MLX, we should implement:**

```python
# mad/blocks/mlstm_mlx/backends.py
class MLXmLSTMBackend:
    def __init__(self, config):
        self.chunkwise_kernel = config.get('chunkwise_kernel', 'native')
        self.step_kernel = config.get('step_kernel', 'native')

    def forward_chunkwise(self, x, state):
        if self.chunkwise_kernel == 'native':
            return self._native_chunkwise(x, state)
        elif self.chunkwise_kernel == 'metal':
            return self._metal_chunkwise(x, state)
        elif self.chunkwise_kernel == 'parallel_scan':
            return self._parallel_scan_chunkwise(x, state)

    def forward_step(self, x, state):
        # Single-token optimized path
        if self.step_kernel == 'native':
            return self._native_step(x, state)
        elif self.step_kernel == 'metal':
            return self._metal_step(x, state)
```

### 2. Per-Head Gating is Critical

```python
# mad/blocks/mlstm_mlx/ffn_block.py
class mLSTMBlockMLX(mx.nn.Module):
    def __init__(self, config):
        # Gates are PER-HEAD (not per-feature!)
        self.igate = mx.nn.Linear(d_model, num_heads, bias=True)
        self.fgate = mx.nn.Linear(d_model, num_heads, bias=True)

        # Output gate is per-feature
        self.ogate = mx.nn.Linear(d_model, d_model, bias=False)
```

### 3. State Management Pattern

```python
# mad/blocks/mlstm_mlx/ffn_block.py
def init_state(self, batch_size):
    """Initialize (C, n, m) state tuple"""
    C_0 = mx.zeros((batch_size, num_heads, head_dim, head_dim))
    n_0 = mx.zeros((batch_size, num_heads, head_dim))
    m_0 = mx.zeros((batch_size, num_heads))  # Running max per head!
    return (C_0, n_0, m_0)
```

### 4. Dual-Path Architecture

```python
# mad/blocks/mlstm_mlx/ffn_block.py
class mLSTMBlockMLX(mx.nn.Module):
    def __call__(self, x, state=None, mode='chunkwise'):
        if mode == 'chunkwise':
            # Full sequence, parallel within chunks
            return self._forward_chunkwise(x, state)
        elif mode == 'step':
            # Single token, optimized path
            return self._forward_step(x, state)
```

## Next Implementation Steps

### Priority 1: Native MLX Implementation

Match the canonical structure exactly:

```python
# mad/blocks/mlstm_mlx/canonical_block.py
class CanonicalMLSTMBlock(mx.nn.Module):
    """Exact port of official xLSTM implementation to MLX"""

    def __init__(self, d_model=512, num_heads=4, head_dim=128):
        # Match official architecture exactly
        self.q_proj = mx.nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = mx.nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = mx.nn.Linear(d_model, d_model, bias=False)

        # Per-head gates (critical!)
        self.igate = mx.nn.Linear(d_model, num_heads, bias=True)
        self.fgate = mx.nn.Linear(d_model, num_heads, bias=True)
        self.ogate = mx.nn.Linear(d_model, d_model, bias=False)

        self.out_proj = mx.nn.Linear(d_model, d_model, bias=False)
        self.multihead_norm = MultiHeadLayerNorm(num_heads, head_dim)
```

### Priority 2: Validate Numerical Parity

```python
# tests/test_mlx_canonical_parity.py
def test_mlx_vs_pytorch():
    # Load official checkpoint
    pytorch_model = load_official_xlstm_checkpoint()

    # Convert to MLX
    mlx_model = convert_checkpoint_to_mlx(pytorch_model)

    # Test forward pass
    input_tokens = mx.array([[1, 2, 3, 4, 5]])

    pytorch_out = pytorch_model(input_tokens)
    mlx_out = mlx_model(input_tokens)

    # Should match within tolerance
    assert mx.allclose(mlx_out, mx.array(pytorch_out), atol=1e-2)
```

### Priority 3: Optimize with Metal Kernels

Once parity is validated, optimize hot paths:

```python
# mad/blocks/mlstm_mlx/kernels/metal_chunkwise.py
def metal_chunkwise_forward(q, k, v, igate, fgate, chunk_size=64):
    """
    Metal-optimized chunkwise mLSTM forward pass

    Uses Metal shaders for:
    - Parallel scan for m_t
    - Fused gate computation
    - Optimized matrix operations
    """
    # Implement using mx.fast.metal_kernel
    ...
```

## Key Takeaways

1. **Three kernel types needed**: chunkwise (training), sequence (inference), step (generation)
2. **Per-head gating**: Input/forget gates are per-head, not per-feature
3. **Pre-normalization**: RMSNorm before each layer (Megatron fix)
4. **State is (C, n, m)**: Three tensors per block, m is the exponential scaling factor
5. **Numerical parity**: Chunkwise and step-by-step produce same results (within 0.01)
6. **Flexible backend**: Can swap kernels for different use cases

The official implementation validates everything we've been discussing about exponential gating, parallel chunking, and
stabilization. Now we just need to port it to MLX with Metal optimization!
