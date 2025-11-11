# mLSTM Chunkwise - Clean NCPS Architecture

A modular, composable implementation of Matrix LSTM (mLSTM) following Neural Circuit Policy (NCPS) design patterns with
Metal-accelerated kernels.

## Architecture Overview

This implementation follows a clean separation of concerns, breaking mLSTM into composable cells that can be wired
together:

```
Input [B, S, D]
    ↓
┌─────────────────────────────────┐
│  mLSTMProjectionCell            │  ← Before Cell
│  • Q/K/V projections            │
│  • Gate projections (i, f)      │
│  • No recurrence                │
└─────────────────────────────────┘
    ↓ q, k, v, i_preact, f_preact
┌─────────────────────────────────┐
│  Kernel Cell (Dispatched)       │  ← During Cell
│                                 │
│  ┌───────────────────────────┐  │
│  │ Parallel (Chunkwise)      │  │  • Metal kernels
│  │  • 2-phase algorithm      │  │  • Training
│  │  • Batch processing       │  │  • 8-55x speedup
│  └───────────────────────────┘  │
│           OR                    │
│  ┌───────────────────────────┐  │
│  │ Recurrent (Sequential)    │  │  • Pure MLX
│  │  • Step-by-step           │  │  • Inference
│  │  • Autoregressive         │  │  • Memory efficient
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓ h [B, NH, S, DH_v], state
┌─────────────────────────────────┐
│  mLSTMOutputCell                │  ← After Cell
│  • MultiHead RMS norm           │
│  • Output gate modulation       │
│  • Final projection             │
│  • No recurrence                │
└─────────────────────────────────┘
    ↓
Output [B, S, D], new_state
```

## Design Principles

### 1. **Modular Cells**

Each cell is an `nn.Module` with a single, clear responsibility:

- **No duplication**: Shared cells (projection, output) used by both kernel modes
- **Testable**: Each cell can be tested independently
- **Composable**: Cells are wired together by the neuron

### 2. **NCPS Pattern**

Following Neural Circuit Policy design:

- **Cells**: Hold trainable parameters, implement specific transformations
- **Neurons**: Wire cells together, handle dispatch logic
- **Blocks**: Higher-level wrappers (will add pre-norm + residuals)
- **Models**: Complete applications

### 3. **Kernel Integration**

Metal-accelerated kernels are isolated in kernel cells:

- Projection/output cells are **kernel-agnostic**
- Kernel cells **only** contain recurrence logic
- Easy to swap kernel implementations without touching other components

## File Structure

```
mlstm_chunkwise/
├── README.md                         # This file
├── __init__.py                       # Module exports
│
├── mlstm_projection_cell.py          # Before Cell
│   └── mLSTMProjectionCell
│       • Q/K/V projections: x → q, k, v
│       • Gate projections: x → i_preact, f_preact
│       • Returns: (q, k, v, i_preact, f_preact)
│
├── mlstm_parallel_kernel_cell.py     # Kernel Cell (Parallel)
│   └── mLSTMParallelKernelCell
│       • Two-phase chunkwise algorithm
│       • Metal kernel integration
│       • Phase 1: Inter-chunk recurrence (sequential)
│       • Phase 2: Intra-chunk outputs (parallel)
│       • Receives: (q, k, v, i_preact, f_preact, state)
│       • Returns: (h, new_state)
│
├── mlstm_recurrent_kernel_cell.py    # Kernel Cell (Sequential)
│   └── mLSTMRecurrentKernelCell
│       • Step-by-step sequential recurrence
│       • Pure MLX implementation
│       • Suitable for inference/autoregressive
│       • Receives: (q, k, v, i_preact, f_preact, state)
│       • Returns: (h, new_state)
│
├── mlstm_output_cell.py              # After Cell
│   └── mLSTMOutputCell
│       • Per-head RMS normalization
│       • Output gate: x_orig → o_gate
│       • Output projection: h_gated → output
│       • Receives: (h, x_orig)
│       • Returns: output
│
├── mlstm_neuron.py                   # Neuron (Wiring)
│   └── mLSTMNeuron
│       • Owns all cells (projection, kernels, output)
│       • Handles kernel dispatch (parallel vs recurrent)
│       • Wires the complete pipeline
│       • Receives: (x, state)
│       • Returns: (output, new_state)
│
└── forward/                          # Metal Kernels
    ├── mlstm_chunkwise_recurrent_fw_C.py
    └── mlstm_chunkwise_parallel_fw_Hintra.py
```

## Metal Kernel Integration

### Two-Phase Chunkwise Algorithm

The parallel kernel uses a sophisticated two-phase algorithm for efficient training:

#### **Phase 1: Recurrent (Inter-chunk)**

```
# Compute states BETWEEN chunks (sequential)
matC_states, vecN_states, scaMinter_states = \
    mlstm_chunkwise_recurrent_fw_C_metal(
        matK, matV, vecF, vecI,
        matC_initial, vecN_initial, scaMinter_initial,
        NC, L, ...
    )
```

- Processes chunk boundaries sequentially
- Maintains causal dependencies across chunks
- Metal-accelerated matrix operations

#### **Phase 2: Parallel (Intra-chunk)**

```
# Compute outputs WITHIN chunks (parallel)
matHout, vecNout, vecMout = \
    mlstm_chunkwise_parallel_fw_Hintra_metal(
        matQ, matK, matV,
        matC_states, vecN_states, scaMinter_states,
        vecI, vecB, NC, L, ...
    )
```

- All chunks processed in parallel
- Uses inter-chunk states from Phase 1
- Massive speedup on Apple Silicon (8-55x)

### Kernel Isolation

**Key Design Decision**: Kernels are ONLY called from kernel cells.

```
# CORRECT: Kernel cells call Metal kernels
class mLSTMParallelKernelCell:
    def __call__(self, q, k, v, i_preact, f_preact, state):
        # Calls Metal kernels here
        h, new_state = mlstm_chunkwise_forward(...)
        return h, new_state

# INCORRECT: Neurons don't call kernels directly
class mLSTMNeuron:
    def __call__(self, x, state):
        # Does NOT call kernels
        # Delegates to kernel cells
        h, new_state = self.parallel_kernel(q, k, v, ...)
```

## Usage Examples

### Basic Usage

```
from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_chunkwise import mLSTMNeuron

# Create neuron (parallel mode for training)
neuron = mLSTMNeuron(
    input_size=512,
    num_heads=8,
    qk_dim_per_head=64,
    v_dim_per_head=128,
    chunk_size=64,
    kernel_mode="parallel",  # or "recurrent"
    use_bias=False,
    eps=1e-6,
)

# Forward pass
x = mx.random.normal((2, 100, 512))  # [batch, seq, features]
output, state = neuron(x)
# output: [2, 100, 512]
# state: (C, n, m) for recurrence
```

### Switching Kernel Modes

```
# Training: Use parallel mode
train_neuron = mLSTMNeuron(
    input_size=512,
    num_heads=8,
    qk_dim_per_head=64,
    v_dim_per_head=128,
    kernel_mode="parallel",  # Fast batch processing
)

# Inference: Use recurrent mode
inference_neuron = mLSTMNeuron(
    input_size=512,
    num_heads=8,
    qk_dim_per_head=64,
    v_dim_per_head=128,
    kernel_mode="recurrent",  # Memory efficient
)

# Same architecture, different execution
```

### Using Individual Cells

```
from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_chunkwise import (
    mLSTMProjectionCell,
    mLSTMParallelKernelCell,
    mLSTMOutputCell,
)

# Create cells separately
projection = mLSTMProjectionCell(512, 8, 64, 128)
kernel = mLSTMParallelKernelCell(8, 64, 128, chunk_size=64)
output = mLSTMOutputCell(512, 8, 128)

# Manual wiring (neuron does this automatically)
q, k, v, i_preact, f_preact = projection(x)
h, new_state = kernel(q, k, v, i_preact, f_preact, state)
result = output(h, x)
```

## State Management

### State Format

```
state = (C, n, m)
# C: Cell state [B, NH, DH_qk, DH_v] - covariance matrix
# n: Normalizer [B, NH, DH_qk] - running sum
# m: Stabilizer [B, NH] - running max (for numerical stability)
```

### State Initialization

```
# Automatic: state=None initializes to zeros
output1, state1 = neuron(x1, state=None)

# Recurrent: pass previous state
output2, state2 = neuron(x2, state=state1)
output3, state3 = neuron(x3, state=state2)
```

## Key Differences from Canonical xLSTM

### Our Implementation (NCPS Pattern)

```
mLSTMNeuron (wiring)
  ├─ mLSTMProjectionCell (before)
  ├─ mLSTMParallelKernelCell (during) [dispatched]
  └─ mLSTMOutputCell (after)
```

### Canonical xLSTM (Monolithic)

```
mLSTMLayer (everything)
  ├─ Q/K/V projections
  ├─ Gate projections
  ├─ Recurrence (embedded)
  ├─ Normalization
  └─ Output projection
```

## Performance

### Parallel Mode (Training)

- **Chunk Size**: 64 (default)
- **Speedup**: 8-55x on Apple Silicon
- **Best For**: Batch training, long sequences
- **Memory**: O(sequence_length)

### Recurrent Mode (Inference)

- **Processing**: Step-by-step
- **Speedup**: N/A (sequential)
- **Best For**: Autoregressive generation, streaming
- **Memory**: O(1) per timestep

## Testing

Each cell can be tested independently:

```
# Test projection cell
projection = mLSTMProjectionCell(512, 8, 64, 128)
x = mx.random.normal((2, 10, 512))
q, k, v, i, f = projection(x)
assert q.shape == (2, 8, 10, 64)  # [B, NH, S, DH_qk]

# Test kernel cell
kernel = mLSTMParallelKernelCell(8, 64, 128)
h, state = kernel(q, k, v, i, f, state=None)
assert h.shape == (2, 8, 10, 128)  # [B, NH, S, DH_v]

# Test output cell
output = mLSTMOutputCell(512, 8, 128)
result = output(h, x)
assert result.shape == (2, 10, 512)  # [B, S, input_size]
```

## Future Extensions

### Adding New Kernel Backends

```
# Create new kernel cell
class mLSTMCUDAKernelCell(nn.Module):
    def __call__(self, q, k, v, i_preact, f_preact, state):
        # CUDA implementation
        h, new_state = cuda_mlstm_kernel(...)
        return h, new_state

# Add to neuron
class mLSTMNeuron:
    def __init__(self, ..., kernel_mode="parallel"):
        # ...
        if kernel_mode == "cuda":
            self.kernel = mLSTMCUDAKernelCell(...)
```

No changes needed to projection/output cells!

### Custom Wiring

```
# Custom neuron with different cell compositions
class CustomMLSTMNeuron(nn.Module):
    def __init__(self):
        self.projection = mLSTMProjectionCell(...)
        self.kernel1 = mLSTMParallelKernelCell(...)
        self.kernel2 = mLSTMRecurrentKernelCell(...)
        self.output = mLSTMOutputCell(...)

    def __call__(self, x, state):
        q, k, v, i, f = self.projection(x)
        # Custom logic: use both kernels
        h1, s1 = self.kernel1(q, k, v, i, f, state)
        h2, s2 = self.kernel2(q, k, v, i, f, state)
        h = (h1 + h2) / 2  # Ensemble
        output = self.output(h, x)
        return output, (s1, s2)
```

## References

- **xLSTM Paper**: [Extended Long Short-Term Memory](https://arxiv.org/pdf/2405.04517)
- **NCPS**: [Neural Circuit Policies](https://github.com/mlech26l/ncps)
- **MLX**: [Apple MLX Framework](https://github.com/ml-explore/mlx)

## Contributing

When adding new features:

1. **Keep cells focused**: Each cell should do ONE thing
2. **Test independently**: Write unit tests for each cell
3. **Document clearly**: Explain inputs/outputs and tensor shapes
4. **Follow patterns**: Use consistent APIs across cells
5. **Isolate kernels**: Only kernel cells call Metal/CUDA code

---

**Built with**: Clean architecture principles, NCPS patterns, and Metal acceleration.
