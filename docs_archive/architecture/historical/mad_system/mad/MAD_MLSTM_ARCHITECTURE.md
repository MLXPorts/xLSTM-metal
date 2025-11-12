# MAD Decomposition of Canonical mLSTMLayer

## Overview

This document defines how to decompose the canonical xLSTM-7B `mLSTMLayer` into MAD atomic blocks with proper wiring.

**Philosophy**: Embrace MAD! Create SMALL, composable, atomic blocks. Wire them together using MAD composition instead
of monolithic classes.

## Canonical mLSTMLayer Data Flow

```
Input: x [B, S, embedding_dim=4096]
  │
  ├─> proj_up: Linear(4096 → 2*5440=10880) ────> [B, S, 10880]
  │                                                      │
  │                                              split (dim=-1)
  │                                                      │
  │                                          ┌───────────┴───────────┐
  │                                          │                       │
  │                                   x_mlstm [5440]             z [5440]
  │                                          │                       │
  │                                   CausalConv1d                   │
  │                                   kernel_size=4                  │
  │                                          │                       │
  │                                       SiLU()                     │
  │                                          │                       │
  │                              ┌───────────┼───────────┐           │
  │                              │           │           │           │
  │                          q_proj      k_proj    (skip to v)       │
  │                       (from SiLU)  (from SiLU) (from x_mlstm)    │
  │                              │           │           │           │
  │                              └─────> mLSTMCell <─────┘           │
  │                                          │                       │
  │                                     h_tilde [5440]               │
  │                                          │                       │
  │                              learnable_skip * SiLU(conv)         │
  │                                          │                       │
  │                                  h_tilde + skip                  │
  │                                          │                       │
  │                                  h_tilde_skip * SiLU(z) <────────┘
  │                                          │
  │                                  proj_down: Linear(5440 → 4096)
  │                                          │
  └─────────────────────────────────────────┴─────> Output [B, S, 4096]
```

## MAD Atomic Blocks

### 1. UpProjectionSplitBlock

**Purpose**: Expand dimension 2x and split into two branches

```python
class UpProjectionSplitBlock:
    """
    Projects input to 2x inner dimension and splits into two equal parts.

    Inputs:
        x: [B, S, embedding_dim]

    Outputs:
        branch_a: [B, S, inner_dim]  # For mLSTM processing
        branch_b: [B, S, inner_dim]  # For output gate
    """
    proj: Linear(embedding_dim → 2*inner_dim)

    def __call__(x):
        projected = proj(x)  # [B, S, 2*inner_dim]
        return mx.split(projected, 2, axis=-1)  # Two [B, S, inner_dim]
```

### 2. CausalConv1dBlock

**Purpose**: Temporal convolution with causal masking + activation

```python
class CausalConv1dBlock:
    """
    Causal 1D convolution with SiLU activation.
    Maintains causality: output[t] only depends on input[<=t].

    Inputs:
        x: [B, S, feature_dim]

    Outputs:
        x_conv_act: [B, S, feature_dim]  # Activated conv output
    """
    conv1d: CausalConv1d(feature_dim, kernel_size=4)
    activation: SiLU()

    def __call__(x):
        return activation(conv1d(x))
```

### 3. mLSTMCoreBlock (Current Implementation)

**Purpose**: Core mLSTM computation with Q/K/V projections + kernels

```python
class mLSTMCoreBlock:
    """
    Core mLSTM processing:
    - Q/K from one input (conv-activated branch)
    - V from another input (pre-conv branch)
    - Input/forget/output gates
    - mLSTM kernel (chunkwise/recurrent)

    Inputs:
        x_qk: [B, S, inner_dim]  # For Q/K projections (post-conv)
        x_v: [B, S, inner_dim]   # For V projection (pre-conv)
        x_gates: [B, S, inner_dim]  # For i/f/o gate projections

    Outputs:
        h: [B, S, inner_dim]  # mLSTM hidden states
    """
    # Current implementation - already correct!
```

### 4. SkipConnectionBlock

**Purpose**: Learnable skip connection (elementwise scale + add)

```python
class SkipConnectionBlock:
    """
    Learnable elementwise skip connection.

    Inputs:
        main: [B, S, dim]  # Main path
        skip: [B, S, dim]  # Skip path

    Outputs:
        out: [B, S, dim]  # main + learnable_scale * skip
    """
    learnable_scale: Parameter([dim])  # Initialized to ones

    def __call__(main, skip):
        return main + learnable_scale * skip
```

### 5. OutputGateBlock

**Purpose**: Elementwise gating with activation

```python
class OutputGateBlock:
    """
    Elementwise gating: x * activation(gate).

    Inputs:
        x: [B, S, dim]     # Main signal
        gate: [B, S, dim]  # Gate signal

    Outputs:
        out: [B, S, dim]  # x * SiLU(gate)
    """
    activation: SiLU()

    def __call__(x, gate):
        return x * activation(gate)
```

### 6. DownProjectionBlock

**Purpose**: Reduce dimension back to model dimension

```python
class DownProjectionBlock:
    """
    Projects from inner dimension back to model dimension.

    Inputs:
        x: [B, S, inner_dim]

    Outputs:
        out: [B, S, embedding_dim]
    """
    proj: Linear(inner_dim → embedding_dim)
    dropout: Dropout(p)

    def __call__(x):
        return dropout(proj(x))
```

## MAD Wiring Strategy

**Canonical mLSTMLayer = Wired composition of atomic blocks**

```python
# Small blocks (atomic operations)
up_proj_split = UpProjectionSplitBlock(embedding_dim=4096, inner_dim=5440)
causal_conv = CausalConv1dBlock(feature_dim=5440, kernel_size=4)
mlstm_core = mLSTMCoreBlock(config)  # Current implementation
skip_conn = SkipConnectionBlock(dim=5440)
output_gate = OutputGateBlock()
down_proj = DownProjectionBlock(inner_dim=5440, embedding_dim=4096)

# Wiring (data flow graph)
def mlstm_layer_wired(x):
    # Step 1: Up-project and split
    x_mlstm, z = up_proj_split(x)

    # Step 2: Causal conv on mlstm branch
    x_conv_act = causal_conv(x_mlstm)

    # Step 3: Core mLSTM (Q/K from conv, V from pre-conv)
    h_tilde = mlstm_core(
        x_qk=x_conv_act,      # Q/K from post-conv
        x_v=x_mlstm,          # V from pre-conv
        x_gates=x             # Gates from original input
    )

    # Step 4: Skip connection
    h_tilde_skip = skip_conn(main=h_tilde, skip=x_conv_act)

    # Step 5: Output gating
    h_gated = output_gate(x=h_tilde_skip, gate=z)

    # Step 6: Down-project
    y = down_proj(h_gated)

    return y
```

## Block Size Strategy

### Small Blocks (Atomic)

- **UpProjectionSplitBlock**: Single Linear + split
- **CausalConv1dBlock**: Conv + activation
- **SkipConnectionBlock**: Scale + add
- **OutputGateBlock**: Activation + multiply
- **DownProjectionBlock**: Linear + dropout

### Large Blocks (Composed)

- **mLSTMLayerBlock**: Wired composition of all small blocks above
- Could create intermediate compositions like:
    - **ConvBranchBlock** = Conv + activation + projections
    - **GatedOutputBlock** = Skip + gate + down-proj

**Guideline**: Start with small atomic blocks. Create larger composed blocks only when the composition pattern is reused
frequently.

## Benefits of MAD Decomposition

1. **Modularity**: Each block is independently testable
2. **Flexibility**: Easy to swap implementations (e.g., different conv kernels)
3. **Composability**: Build complex architectures from simple blocks
4. **Debugging**: Clear data flow, easy to inspect intermediate outputs
5. **Reusability**: Atomic blocks can be reused in different architectures

## Next Steps

1. Implement atomic blocks as separate MAD blocks
2. Create wiring configuration for canonical mLSTMLayer
3. Test wired composition matches canonical output
4. Extend to sLSTM and other variants
