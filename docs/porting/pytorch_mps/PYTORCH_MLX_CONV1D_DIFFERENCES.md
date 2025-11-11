# PyTorch vs MLX Conv1d Format Differences

## Critical Finding: Channel Dimension Position

### PyTorch Conv1d

**Input Shape**: `(N, C_in, L)`

- N = Batch size
- C_in = Number of input channels
- L = Sequence length

**Channels are FIRST** (after batch dimension)

```python
import torch.nn as nn

# PyTorch Conv1d
conv = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4)
x_torch = torch.randn(32, 256, 100)  # (batch=32, channels=256, length=100)
out = conv(x_torch)  # (32, 512, 97) with padding=0
```

### MLX Conv1d

**Input Shape**: `(N, L, C)`

- N = Batch size
- L = Sequence length
- C = Number of input channels

**Channels are LAST**

```python
import mlx.nn as nn

# MLX Conv1d
conv = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4)
x_mlx = mx.random.normal((32, 100, 256))  # (batch=32, length=100, channels=256)
out = conv(x_mlx)  # (32, 97, 512) with padding=0
```

## Canonical xLSTM Implementation

The canonical `CausalConv1d` in xlstm package uses **PyTorch format (NCL)**:

```python
# From xlstm/components/conv.py
class CausalConv1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, feature_dim)
        # BUT internally reshapes to (B, feature_dim, S) for PyTorch Conv1d!

        x = x.transpose(1, 2)  # (B, S, C) -> (B, C, S)
        x = self.conv1d(x)     # PyTorch Conv1d expects (B, C, S)
        x = x.transpose(1, 2)  # (B, C, S) -> (B, S, C)
        return x
```

So the canonical implementation:

1. Accepts input as `(B, S, C)` (sequence-first, channels-last)
2. Transposes to `(B, C, S)` for PyTorch Conv1d
3. Transposes back to `(B, S, C)`

## MLX MAD Implementation Strategy

For MLX, we have two choices:

### Option 1: Match Canonical API (Recommended)

Accept `(B, S, C)` input, use MLX Conv1d directly (no transpose needed!)

```python
class CausalConv1dBlock:
    """MLX-native causal conv - no transpose needed!"""
    def __init__(self, feature_dim, kernel_size):
        # MLX Conv1d natively expects (N, L, C) format
        self.conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1  # Causal padding
        )

    def __call__(self, x):
        # x: (B, S, feature_dim) - already in MLX format!
        x = self.conv(x)  # No transpose needed

        # Causal cropping: remove future-looking positions
        if x.shape[1] > S:
            x = x[:, :-(kernel_size-1), :]

        return x
```

### Option 2: Explicit Transpose (Unnecessary)

Would require transpose to match PyTorch format, then transpose back - **wasteful!**

## Recommendation

**Use Option 1**: MLX Conv1d already expects the format canonical xLSTM uses after transpose. We can skip the transpose
entirely and use MLX Conv1d directly on `(B, S, C)` tensors!

This is more efficient than the canonical implementation since we avoid unnecessary transposes.

## Implementation Notes

1. **Causal Padding**: PyTorch uses left-padding for causal conv. MLX pads symmetrically, so we need to:
    - Pad left with `kernel_size - 1`
    - Crop output by `kernel_size - 1` from the right

2. **Stateful Conv (for generation)**: Need to maintain conv state for single-token generation:
   ```python
   def step(self, x_t, conv_state):
       # x_t: (B, 1, C)
       # conv_state: (B, kernel_size-1, C) - previous tokens

       # Concat with state
       x_with_context = mx.concatenate([conv_state, x_t], axis=1)

       # Apply conv
       out = self.conv(x_with_context)[:, -1:, :]  # Take last position

       # Update state (rolling window)
       new_state = x_with_context[:, 1:, :]  # Drop oldest

       return out, new_state
   ```

## Summary

- **PyTorch**: `(N, C, L)` - Channels first
- **MLX**: `(N, L, C)` - Channels last
- **Canonical xLSTM**: Transposes to/from PyTorch format
- **MLX MAD**: Can use MLX Conv1d directly without transpose - **more efficient!**
