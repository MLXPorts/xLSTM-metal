# xLSTM Architecture Sequences

**Canonical block orderings for MAD evaluation and numerical validation**

Date: 2025-01-21
Status: Specification
Based on: xLSTM paper (Beck et al., 2024), xlstm-large checkpoint

## Overview

xLSTM architectures use a **signature** pattern `(m, s)` where:

- `m` = number of mLSTM blocks
- `s` = number of sLSTM blocks
- Blocks repeat in this pattern throughout the architecture

Each block consists of:

1. **Sequence mixer** (mLSTM or sLSTM)
2. **Channel mixer** (GatedFFN / SwiGLU)

## Canonical Patterns

### Pattern 1: xLSTM[7:1] (xlstm-large)

**Signature:** `(7, 1)` — 7 mLSTM blocks + 1 sLSTM block

**Block sequence:**

```
[mLSTM + FFN] × 7
[sLSTM + FFN] × 1
```

**Full layer sequence** (16 layers per 8-block cycle):

```python
layer_types = [
    # Block 1 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 0: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 1: FFN

    # Block 2 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 2: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 3: FFN

    # Block 3 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 4: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 5: FFN

    # Block 4 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 6: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 7: FFN

    # Block 5 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 8: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 9: FFN

    # Block 6 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 10: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 11: FFN

    # Block 7 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 12: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 13: FFN

    # Block 8 (sLSTM)
    {'type': 'slstm', 'backend': 'mlx'},      # Layer 14: sLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 15: FFN

    # Repeat pattern for remaining blocks...
]
```

**xlstm-large specs:**

- Total blocks: 48 (= 6 cycles × 8 blocks/cycle)
- Total layers: 96 (= 48 blocks × 2 layers/block)
- mLSTM blocks: 42 (= 7/8 × 48)
- sLSTM blocks: 6 (= 1/8 × 48)
- Embedding dim: 2048
- Num heads: 8
- Head dim: 256
- FFN factor: 2.6667

### Pattern 2: xLSTM[1:1] (Balanced)

**Signature:** `(1, 1)` — Alternating mLSTM and sLSTM

**Block sequence:**

```
[mLSTM + FFN] × 1
[sLSTM + FFN] × 1
```

**Full layer sequence** (4 layers per 2-block cycle):

```python
layer_types = [
    # Block 1 (mLSTM)
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 0: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 1: FFN

    # Block 2 (sLSTM)
    {'type': 'slstm', 'backend': 'mlx'},      # Layer 2: sLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 3: FFN

    # Repeat pattern...
]
```

**Use case:** Balanced evaluation of mLSTM vs sLSTM

### Pattern 3: xLSTM[1:0] (mLSTM-only)

**Signature:** `(1, 0)` — Only mLSTM blocks

**Block sequence:**

```
[mLSTM + FFN] × repeat
```

**Full layer sequence:**

```python
layer_types = [
    # Block 1
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 0: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 1: FFN

    # Block 2
    {'type': 'mlstm', 'backend': 'mlx'},      # Layer 2: mLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 3: FFN

    # Repeat...
]
```

**Use case:** Baseline for MAD evaluation (current implementation)

### Pattern 4: xLSTM[0:1] (sLSTM-only)

**Signature:** `(0, 1)` — Only sLSTM blocks

**Block sequence:**

```
[sLSTM + FFN] × repeat
```

**Full layer sequence:**

```python
layer_types = [
    # Block 1
    {'type': 'slstm', 'backend': 'mlx'},      # Layer 0: sLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 1: FFN

    # Block 2
    {'type': 'slstm', 'backend': 'mlx'},      # Layer 2: sLSTM
    {'type': 'swiglu', 'backend': 'mlx'},     # Layer 3: FFN

    # Repeat...
]
```

**Use case:** Ablation study for sLSTM

## MAD Multi-Backend Sequences

### Hybrid Backend: MLX + PyTorch

Test backend interoperability:

```python
layer_types = [
    # Block 1: MLX mLSTM
    {'type': 'mlstm', 'backend': 'mlx'},
    {'type': 'swiglu', 'backend': 'mlx'},

    # Block 2: PyTorch chunkwise mLSTM
    {'type': 'chunkwise-mlstm', 'backend': 'pytorch', 'params': {'chunk_size': 64}},
    {'type': 'swiglu', 'backend': 'pytorch'},

    # Block 3: torch.compile mLSTM (Metal MPS)
    {'type': 'torch-compiled-mlstm', 'backend': 'pytorch'},
    {'type': 'swiglu', 'backend': 'pytorch'},

    # Repeat...
]
```

**Use case:** Validate numerical parity across backends

### MAD Architecture Zoo

Test different sequence mixers from MAD registry:

```python
layer_types = [
    # Block 1: mLSTM
    {'type': 'mlstm', 'backend': 'mlx'},
    {'type': 'swiglu', 'backend': 'mlx'},

    # Block 2: Attention
    {'type': 'attention', 'backend': 'mlx', 'params': {'num_heads': 8}},
    {'type': 'swiglu', 'backend': 'mlx'},

    # Block 3: Hyena
    {'type': 'hyena', 'backend': 'pytorch'},
    {'type': 'swiglu', 'backend': 'pytorch'},

    # Block 4: Mamba
    {'type': 'mamba', 'backend': 'pytorch'},
    {'type': 'swiglu', 'backend': 'pytorch'},

    # Repeat...
]
```

**Use case:** MAD's mechanistic architecture evaluation

## Configuration Generation

### Python Helper Function

```python
def generate_xlstm_sequence(
    signature: tuple[int, int],
    num_cycles: int,
    backend: str = 'mlx',
    mlstm_params: dict = None,
    slstm_params: dict = None,
    ffn_params: dict = None,
) -> list[dict]:
    """
    Generate xLSTM layer sequence from signature.

    Args:
        signature: (m, s) where m=num mLSTM blocks, s=num sLSTM blocks
        num_cycles: Number of times to repeat the signature pattern
        backend: Default backend ('mlx', 'pytorch', etc.)
        mlstm_params: Parameters for mLSTM layers
        slstm_params: Parameters for sLSTM layers
        ffn_params: Parameters for FFN layers

    Returns:
        List of layer specifications

    Example:
        >>> generate_xlstm_sequence((7, 1), num_cycles=6)
        # Returns 96-layer sequence for xlstm-large (48 blocks)
    """
    m, s = signature
    mlstm_params = mlstm_params or {}
    slstm_params = slstm_params or {}
    ffn_params = ffn_params or {}

    layer_types = []

    for _ in range(num_cycles):
        # mLSTM blocks
        for _ in range(m):
            layer_types.append({
                'type': 'mlstm',
                'backend': backend,
                'params': mlstm_params.copy()
            })
            layer_types.append({
                'type': 'swiglu',
                'backend': backend,
                'params': ffn_params.copy()
            })

        # sLSTM blocks
        for _ in range(s):
            layer_types.append({
                'type': 'slstm',
                'backend': backend,
                'params': slstm_params.copy()
            })
            layer_types.append({
                'type': 'swiglu',
                'backend': backend,
                'params': ffn_params.copy()
            })

    return layer_types
```

### Usage Examples

```python
# xlstm-large (7:1 pattern, 48 blocks)
xlstm_large_sequence = generate_xlstm_sequence(
    signature=(7, 1),
    num_cycles=6,
    backend='mlx',
    mlstm_params={'num_heads': 8, 'head_dim': 256},
    slstm_params={'num_heads': 8, 'head_dim': 256},
)

# Balanced (1:1 pattern, 24 blocks)
balanced_sequence = generate_xlstm_sequence(
    signature=(1, 1),
    num_cycles=12,
    backend='mlx',
)

# mLSTM-only (1:0 pattern, 12 blocks)
mlstm_only_sequence = generate_xlstm_sequence(
    signature=(1, 0),
    num_cycles=12,
    backend='mlx',
)
```

## ISO-State Normalization

**Critical:** All architectures must use **iso-state normalization** for fair comparison in MAD evaluation.

```python
# For MAD fair comparison
import math

total_state_dim = 4096  # Fixed memory budget
num_heads = 8

head_dim = int(math.sqrt(total_state_dim / num_heads))
# Result: head_dim = 22 (for 8 heads, total state ≈ 4096)

inner_dim = num_heads * head_dim
```

This ensures all architectures have the same state memory footprint, enabling fair MAD task performance comparison.

## Testing Plan

### Test 1: Numerical Parity (xlstm-large)

Load the xlstm-large checkpoint and validate output:

```python
config = {
    'layer_types': generate_xlstm_sequence((7, 1), num_cycles=6, backend='mlx'),
    'd_model': 2048,
    'num_heads': 8,
    'head_dim': 256,
    'vocab_size': 50257,
}

model = MADLanguageModel(config)
model.load_state_dict(checkpoint['model'])

# Test forward pass
x = torch.randint(0, vocab_size, (2, 128))
logits, state = model(x)

# Compare with reference implementation
assert torch.allclose(logits, reference_logits, atol=1e-4)
```

### Test 2: Sequence Length Scaling

Test all canonical patterns at multiple sequence lengths:

| Pattern | Seq Len | Batch Size | Expected Behavior |
|---------|---------|------------|-------------------|
| 7:1     | 128     | 32         | Baseline          |
| 7:1     | 512     | 16         | Linear scaling    |
| 7:1     | 2048    | 4          | Long context      |
| 7:1     | 8192    | 1          | Max context       |

### Test 3: Backend Comparison

Compare outputs across backends:

```python
backends = ['mlx', 'pytorch', 'torch_compile']

for backend in backends:
    config['layer_types'] = update_backend(config['layer_types'], backend)
    model = MADLanguageModel(config)

    logits, _ = model(x)
    outputs[backend] = logits

# Validate numerical parity
for b1, b2 in itertools.combinations(backends, 2):
    assert torch.allclose(outputs[b1], outputs[b2], atol=1e-3), \
        f"Outputs differ: {b1} vs {b2}"
```

### Test 4: MAD Synthetic Tasks

Evaluate all patterns on MAD's 6 tasks:

```python
patterns = {
    '7:1': generate_xlstm_sequence((7, 1), 6),
    '1:1': generate_xlstm_sequence((1, 1), 12),
    '1:0': generate_xlstm_sequence((1, 0), 12),
    '0:1': generate_xlstm_sequence((0, 1), 12),
}

tasks = [
    'in-context-recall',
    'noisy-in-context-recall',
    'fuzzy-in-context-recall',
    'memorization',
    'compression',
    'selective-copying',
]

for pattern_name, layer_types in patterns.items():
    for task_name in tasks:
        score = evaluate_mad_task(layer_types, task_name)
        results[pattern_name][task_name] = score
```

## References

- **xLSTM paper**: Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024
- **xlstm-large checkpoint**: Official pretrained model (2048 dim, 48 blocks, 7:1 pattern)
- **MAD paper**: Mechanistic Architecture Design framework
- **Code**: `src/xlstm_solace_torch/api.py:23` (signature parameter)
- **Config**: `src/xlstm_solace_torch/xlstm_large/config.py`

---

## Appendix: Block-Level vs Layer-Level

**Important distinction:**

- **Block** = Sequence mixer + Channel mixer (e.g., mLSTM + FFN)
- **Layer** = Single operation (e.g., mLSTM **or** FFN)

**xLSTM signature counts BLOCKS, not layers.**

Example: signature `(7, 1)` means:

- 7 blocks = 14 layers (7 mLSTM + 7 FFN)
- 1 block = 2 layers (1 sLSTM + 1 FFN)
- **Total: 8 blocks = 16 layers**

When configuring MAD's `layer_types`, we specify at the **layer level** (not block level), so we need to expand blocks
into their constituent layers.
