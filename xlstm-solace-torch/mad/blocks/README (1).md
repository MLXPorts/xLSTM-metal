# MAD Framework Blocks

xLSTM block implementations for the MAD (Mechanistic Architecture Design) framework, including support for SWAX hybrid architectures (Cabannes et al., Meta FAIR & JKU Linz, 2025).

## Design Principles

- Direct PyTorch implementations
- MAD framework patterns and conventions
- Self-contained, composable blocks
- Compatible with SWAX hybrid architectures

## Available Blocks

### 1. `mLSTMBlock` - Matrix LSTM for Sequence Mixing

Implementation of mLSTM (Extended Long Short-Term Memory) for the MAD framework.

**Features:**
- Matrix memory state (covariance-based updates)
- Exponential gating for numerical stability
- Multi-head architecture
- Optional causal convolution
- Learnable skip connections
- Group normalization for stability

**Usage:**
```python
from xlstm_solace_torch.mad.blocks import mLSTMBlock

block = mLSTMBlock(
    dim=256,
    num_heads=4,
    proj_factor=2.0,
    conv_kernel_size=4,
    num_blocks=12,  # For weight init scaling
)

# Forward pass
x = torch.randn(2, 64, 256)  # [B, S, D]
y = block(x)  # [B, S, D]
```

**Architecture:**
```
Input [B, S, D]
  ↓
Proj Up: D → 2×inner_dim
  ↓
Split: mlstm_branch | output_gate
  ↓                    ↓
Causal Conv         (bypass)
  ↓
QKV Projections
  ↓
mLSTM Cells (per-head)
  ↓
Concat Heads
  ↓
Learnable Skip
  ↓
Output Gate (element-wise ×)
  ↓
Proj Down: inner_dim → D
  ↓
Output [B, S, D]
```

### 2. `GatedFFN` - Feed-Forward for Channel Mixing

SwiGLU-style gated feed-forward network.

**Features:**
- Gated activation (SwiGLU pattern)
- Configurable expansion factor
- Inner dimension rounding
- Dropout support

**Usage:**
```python
from xlstm_solace_torch.mad.blocks import GatedFFN

ffn = GatedFFN(
    dim=256,
    proj_factor=2.667,  # Canonical xLSTM default
    round_to=64,
    num_blocks=12,
)

x = torch.randn(2, 64, 256)
y = ffn(x)
```

**Architecture:**
```
Input [B, S, D]
  ↓
Parallel Projections
  ↓         ↓
Gate      Up
 ↓         ↓
SiLU   (linear)
  ↓         ↓
  Gate × Up
      ↓
  Proj Down: inner_dim → D
      ↓
  Output [B, S, D]
```

### 3. `SlidingWindowAttention` - Efficient Local Attention (SWAX)

Sliding window attention for SWAX hybrid architectures.

**Features:**
- O(w) complexity per token (w = window size)
- RoPE (Rotary Position Embeddings)
- Multi-head attention
- Stochastic window size support (for training)
- Causal masking

**Usage:**
```python
from xlstm_solace_torch.mad.blocks import SlidingWindowAttention

swa = SlidingWindowAttention(
    dim=256,
    num_heads=16,  # Canonical SWAX uses 16
    window_size=2048,
    num_blocks=12,
)

x = torch.randn(2, 128, 256)

# Regular forward
y = swa(x)

# Stochastic training (override window size)
y = swa(x, window_size=128)
```

## Building SWAX Hybrid Models

Combine mLSTM and SWA blocks in 1:1 ratio (per SWAX paper):

```python
import torch.nn as nn
from xlstm_solace_torch.mad.blocks import mLSTMBlock, GatedFFN, SWABlock

class SWAXModel(nn.Module):
    def __init__(self, dim=2048, num_blocks=24):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Alternate mLSTM and SWA (1:1 ratio)
            if i % 2 == 0:
                layer = mLSTMBlock(dim=dim, num_blocks=num_blocks*2)
            else:
                layer = SWABlock(dim=dim, num_blocks=num_blocks*2)

            # Add pre-norm and post-norm + FFN
            self.blocks.append(nn.ModuleDict({
                'pre_norm': nn.LayerNorm(dim),
                'layer': layer,
                'post_norm': nn.LayerNorm(dim),
                'ffn': GatedFFN(dim=dim, num_blocks=num_blocks*2),
            }))

    def forward(self, x, window_size=None):
        for block in self.blocks:
            # Pre-norm + layer + residual
            h = x + block['layer'](block['pre_norm'](x), window_size=window_size)
            # Post-norm + FFN + residual
            x = h + block['ffn'](block['post_norm'](h))
        return x
```

## SWAX Training Strategy

Per the SWAX paper (Cabannes et al., 2025):

**1.4B model:**
- Default window: 2048
- Stochastic training: p=0.5 chance of w=128
- Last 10% of training: Fixed w=2048 (annealing)

**7B model:**
- Default window: 2048
- Stochastic training: p=0.75 chance of w=128
- Last 10% of training: Fixed w=2048 (annealing)

**Example:**
```python
import random

model = SWAXModel(dim=2048, num_blocks=24)
model.train()

for step in range(num_steps):
    # Stochastic window size
    if step < num_steps * 0.9:  # First 90%
        window_size = 128 if random.random() < 0.5 else 2048
    else:  # Last 10% (annealing)
        window_size = 2048

    logits = model(input_ids, window_size=window_size)
    loss = compute_loss(logits, targets)
    loss.backward()
```

## Key Insights from SWAX Paper

1. **Short windows (128) → Better long-context performance**
   - Forces mLSTM to learn long-term memory
   - Reduces over-reliance on attention for recall

2. **Long windows (2048) → Better short-context performance**
   - Provides high-precision local reasoning
   - Better for reasoning benchmarks

3. **Stochastic training → Best of both worlds**
   - Combines long-context memory with short-context reasoning
   - Acts as a form of dropout on the attention mechanism

4. **1:1 ratio (mLSTM:SWA) works best**
   - Natural split: SWA for local, mLSTM for global
   - Each component specializes in its strength

## Weight Initialization

All blocks use canonical xLSTM initialization:

- **Input projections**: `small_init` (std = √(2/(5×dim)))
- **Output projections**: `wang_init` (std = 2/(num_blocks×√dim))
- **Forget gate bias**: Linspace from 3.0 to 6.0 (encourages retention)

## Tests

Run tests:
```bash
python tests/test_mad_blocks.py
```

All tests should pass (8/8).

## References

- **xLSTM**: Beck et al. (2024) - Extended Long Short-Term Memory
- **SWAX**: Cabannes et al. (2025) - Short window attention enables long-term memorization
- **MAD Framework**: Mechanistic Architecture Design patterns

## Directory Structure

```
mad/blocks/
├── README.md              # This file
├── __init__.py            # Exports all blocks
├── mlstm_cell.py          # mLSTM cell
├── mlstm_block.py         # mLSTM block
├── ffn_block.py           # Gated FFN block
└── swa_block.py           # Sliding window attention block
```

## Implementation Characteristics

1. **Direct implementations** - No adapter layers or wrappers
2. **PyTorch only** - No external dependencies beyond torch
3. **MAD framework patterns** - Follows MAD conventions
4. **Composable** - Each block is self-contained and independent
5. **Numerically stable** - Exponential gating, normalization layers
6. **Production-ready** - Fully implemented, tested, documented
