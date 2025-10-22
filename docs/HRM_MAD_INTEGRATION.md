# HRM+ Integration with MAD Framework

Complete integration of HRM+ (Hierarchical Retrieval Memory) research components with the MAD (Modular Atomically-wired Differentiable) framework for xLSTM-7B.

## Overview

This integration brings together two cutting-edge architectures:

1. **xLSTM-7B**: Production-ready 7B parameter model with mLSTM (matricial LSTM) blocks and optimized Metal kernels
2. **HRM+**: Research architecture featuring:
   - Content-addressable Memory Cubes
   - Liquid Neural Networks (LTC cells)
   - Adaptive Computation Time (ACT) halting
   - Z5 Temporal Discretization
   - Neuromodulation (5-HT serotonin)

## Architecture

### Two-Timescale Design

```
Fast Timescale (mLSTM):
  - Parallel chunk processing
  - Matrix memory and gating
  - Every token

Slow Timescale (Memory Cubes):
  - Content-addressable retrieval
  - Residual predictions
  - Z5 boundary commits (every 5th step)

Optional (Liquid Cells):
  - Continuous-time dynamics
  - Adaptive time constants
  - Per-token recurrence
```

### Integration Strategies

The `create_hrm_xlstm_7b_wiring()` function supports four strategies:

#### 1. **none** - Standard xLSTM
```
embedding → xlstm_0 → ... → xlstm_31 → out_norm → lm_head
```
No HRM augmentation, baseline performance.

#### 2. **per_block** - Fully Enhanced
```
embedding → hrm_xlstm_0 → hrm_xlstm_1 → ... → hrm_xlstm_31 → out_norm → lm_head
```
Every block has its own memory cube. Most parameters, highest capacity.

#### 3. **per_segment** - Hierarchical (Recommended)
```
embedding → [xlstm_0...7 → hrm_gate_0] → [xlstm_8...15 → hrm_gate_1] → ... → out_norm → lm_head
```
Memory gates applied every N blocks. Good balance of capacity and efficiency.

#### 4. **post_process** - Output Enhancement
```
embedding → xlstm_0 → ... → xlstm_31 → hrm_gate → out_norm → lm_head
```
Single HRM wrapper at end. Minimal overhead, good for experimentation.

## Components

### Core HRM+ Blocks (xlstm_metal/blocks/hrm_mlx/)

#### MemoryCubeMLX
Content-addressable memory with cosine similarity retrieval.

```python
cube = MemoryCubeMLX(d_key=512, d_val=512, max_items=65536, topk=8)
pred, conf = cube.query(query_keys)  # Retrieve predictions
cube.update(new_keys, new_values)    # Store new memories
```

**Features:**
- Ring buffer eviction (keeps most recent max_items)
- Top-k weighted retrieval
- Confidence scores from cosine similarities

#### CubeGatedBlockMLX
Memory-augmented blending with learned gates.

```python
block = CubeGatedBlockMLX(d_in=512, fuse_phase_keys=True)
y_out, alpha_mean, conf_mean = block(
    h_in=activations,
    times=time_steps,     # Z5 scheduler
    mod_5ht=serotonin,    # Optional neuromodulation
    allow_commit=commits   # Z5 boundary mask
)
```

**Features:**
- Learned α gate blends input with cube predictions
- Phase-key fusion (cos/sin + Z5 one-hot encoding)
- 5-HT neuromodulation (divisive gain)
- Z5-controlled updates (only commit on slot==4)

#### LiquidTimeConstantMLX
Continuous-time recurrent cell with adaptive dynamics.

```python
cell = LiquidTimeConstantMLX(input_size=512, hidden_size=512)
h_new, output = cell(x, h_prev, t)
```

**Features:**
- Learned time constants τ (via softplus)
- Gated short/long-term blending
- Residual clamping for stability

#### ACTHaltingHeadMLX
Adaptive Computation Time telemetry.

```python
head = ACTHaltingHeadMLX(d_model=512, threshold=0.5)
probs, mask, stats = head(hidden_states)
```

**Features:**
- Per-token halting probabilities
- Threshold-based masking
- Statistics (mean prob, open rate)

#### Z5 Scheduler
Base-5 temporal discretization for boundary commits.

```python
from xlstm_metal.blocks.hrm_mlx import z5_slots, boundary_commit_mask

slots = z5_slots(times)            # Returns {0,1,2,3,4}
commits = boundary_commit_mask(times)  # True when slot==4
```

**Purpose:**
- Enforces carry structure for temporal patterns
- Prevents premature memory updates
- Creates hierarchical time representation

### Composite Block

#### HRMxLSTMBlockMLX
Wraps standard xLSTM block with HRM features.

```python
from xlstm_metal.blocks.mlstm_mlx.xlstm_block import xLSTMBlockConfig
from xlstm_metal.blocks.hrm_mlx import HRMxLSTMConfig, HRMxLSTMBlockMLX

xlstm_cfg = xLSTMBlockConfig(embedding_dim=4096, num_heads=8)
hrm_cfg = HRMxLSTMConfig(
    xlstm_config=xlstm_cfg,
    enable_hrm=True,
    enable_act=True,
    fuse_phase_keys=True
)

block = HRMxLSTMBlockMLX(hrm_cfg)
output, state, telemetry = block(x, times=times)
```

**Telemetry:**
- `alpha_mean`: Gate activation strength
- `conf_mean`: Cube retrieval confidence
- `energy_pre_gate`, `energy_post_gate`: L2 norms
- `act_prob_mean`, `act_open_rate`: Halting metrics

## MAD Wiring Extensions

### New BlockTypes

Extended `xlstm_metal/wiring/core.py`:

```python
class BlockType(Enum):
    # ... existing types ...
    MEMORY_CUBE = "memory_cube"
    LIQUID_CELL = "liquid_cell"
    CUBE_GATED = "cube_gated"
    ACT_HALTING = "act_halting"
    HRM_XLSTM = "hrm_xlstm"
```

### Block Instantiation

Extended `xlstm_metal/wiring/mlx/wiring.py` to instantiate HRM blocks in `WiredMADModel._instantiate_block()`.

### Wiring Helper

Created `xlstm_metal/wiring/mlx/hrm_wiring.py`:

```python
from xlstm_metal.wiring.mlx import create_hrm_xlstm_7b_wiring, WiredMADModel

wiring = create_hrm_xlstm_7b_wiring(
    embedding_dim=4096,
    num_blocks=32,
    hrm_strategy="per_segment",
    hrm_segment_size=8,
    enable_act=True,
    fuse_phase_keys=True,
    cube_max_items=65536,
    cube_topk=8,
    k_5ht=0.5,
    act_threshold=0.5
)

model = WiredMADModel(wiring, 'embedding', 'lm_head')
```

## Usage Examples

### Basic Forward Pass

```python
import mlx.core as mx
from xlstm_metal.wiring.mlx import create_hrm_xlstm_7b_wiring, WiredMADModel

# Create model
wiring = create_hrm_xlstm_7b_wiring(hrm_strategy="per_segment")
model = WiredMADModel(wiring, 'embedding', 'lm_head')

# Prepare input
batch_size, seq_len = 2, 16
input_ids = mx.random.randint(0, 50304, (batch_size, seq_len))

# Forward pass
output, hidden_states = model(input_ids)
# output: (B, L, vocab_size) logits
# hidden_states: dict of per-block states
```

### With Temporal Information

```python
# Generate time steps for Z5 scheduler
times = mx.arange(seq_len).reshape(1, -1)
times = mx.broadcast_to(times, (batch_size, seq_len))

# Note: Current WiredMADModel doesn't thread times through automatically
# Use HRMxLSTMBlockMLX directly for full telemetry:

from xlstm_metal.blocks.hrm_mlx import HRMxLSTMBlockMLX, HRMxLSTMConfig
from xlstm_metal.blocks.mlstm_mlx.xlstm_block import xLSTMBlockConfig

xlstm_cfg = xLSTMBlockConfig(embedding_dim=512, num_heads=8)
hrm_cfg = HRMxLSTMConfig(xlstm_config=xlstm_cfg, enable_hrm=True)
block = HRMxLSTMBlockMLX(hrm_cfg)

x = mx.random.normal((batch_size, seq_len, 512))
output, state, telemetry = block(x, times=times)

print(f"Alpha: {telemetry['alpha_mean']:.3f}")
print(f"Confidence: {telemetry['conf_mean']:.3f}")
print(f"Energy ratio: {telemetry['energy_post_gate'] / telemetry['energy_pre_gate']:.3f}")
```

### With Neuromodulation

```python
# 5-HT serotonin levels (0.0 = low, 1.0 = high)
mod_5ht = mx.ones((batch_size, seq_len)) * 0.5

output, state, telemetry = block(x, times=times, mod_5ht=mod_5ht)

# Higher 5-HT → lower gain → reduced memory influence
# Useful for exploring attention/memory tradeoffs
```

### Comparing Strategies

```python
strategies = ["none", "per_segment", "post_process"]

for strategy in strategies:
    wiring = create_hrm_xlstm_7b_wiring(hrm_strategy=strategy)
    model = WiredMADModel(wiring, 'embedding', 'lm_head')

    output, _ = model(input_ids)
    # Compare outputs, parameter counts, inference speed
```

## Demo Script

Run `examples/hrm_xlstm_demo.py` for comprehensive demonstrations:

```bash
python examples/hrm_xlstm_demo.py
```

Demonstrates:
1. All HRM integration strategies
2. Telemetry collection
3. Neuromodulation effects
4. Z5 scheduler behavior

## Key Design Decisions

### 1. State Management

**Challenge:** Memory cubes are stateful across forward passes.

**Solution:** Cubes maintain internal state (`self.keys`, `self.vals`). Not passed through wiring. Updates happen in-place during `cube.update()`.

**Tradeoff:** Simpler API but requires careful handling of training vs inference.

### 2. Z5 Boundary Commits

**Challenge:** When to update memory cubes?

**Solution:** Z5 scheduler provides `allow_commit` mask. Cubes only update when `slot == 4`.

**Benefit:** Enforces temporal structure, prevents noisy updates.

### 3. Phase-Key Fusion

**Challenge:** How to encode temporal context in memory keys?

**Solution:** Concatenate [key || cos/sin phases || Z5 one-hot] and project.

**Components:**
- 6 trigonometric features (3 cos + 3 sin, periods 1/3/9)
- 5 Z5 slot one-hot
- Total: 11 → project to 8 → concat with key → final projection

### 4. Integration with mLSTM

**Challenge:** LTC cells need per-token recurrence, mLSTM is chunk-parallel.

**Solution:** Apply HRM *after* mLSTM blocks, not within. Memory cubes blend residuals post-hoc.

**Future Work:** Explore liquid-mLSTM hybrids with mixed recurrence/parallelism.

### 5. Telemetry Collection

**Challenge:** Need metrics without breaking forward pass.

**Solution:** HRM blocks return (output, state, telemetry_dict). Wiring needs extension to aggregate telemetry.

**Current Status:** Telemetry works for direct block usage. Full wiring telemetry is future work.

## Performance Considerations

### Memory Usage

- **Per-block HRM:** 32 cubes × 65k items × (key_dim + val_dim) ≈ 16GB for 4096-dim
- **Per-segment (8 blocks):** 4 cubes ≈ 2GB
- **Post-process:** 1 cube ≈ 512MB

**Recommendation:** Use per-segment with hrm_segment_size=8 for 7B model.

### Compute Overhead

- Cube query: Top-k cosine similarity (fast with MLX)
- Phase-key fusion: 6 trig + 5 one-hot + 2 projections (minimal)
- ACT halting: Single linear projection (negligible)

**Estimate:** ~5-10% overhead for per-segment HRM vs standard xLSTM.

### MLX Optimizations

All HRM components use MLX primitives:
- `mx.normalize` for cosine similarity
- `mx.argpartition` for top-k
- `mx.einsum` for weighted averaging
- `mx.sigmoid`, `mx.tanh` for activations

No custom kernels needed - fully graph-compilable.

## Training Considerations

### Ponder Loss (ACT)

ACT halting requires ponder loss during training:

```python
# Collect halting probs from all blocks
total_ponder = 0.0
for block in model.blocks:
    if hasattr(block, 'act_head'):
        probs, _, _ = block.act_head(activations)
        total_ponder += probs.sum()

# Add ponder penalty
ponder_loss = lambda_p * total_ponder / (batch_size * seq_len * num_blocks)
total_loss = ce_loss + ponder_loss
```

### Cube Update Strategy

**Training:**
- Use `train=True` flag
- Provide `allow_commit` mask from Z5 scheduler
- Teacher signal: previous layer output or targets

**Inference:**
- Use `train=False` flag
- Cubes still query but don't update
- Or: update with own predictions (self-supervised)

### Gradient Handling

Memory cubes use `@mx.no_grad()` (equivalent to PyTorch's `@torch.no_grad()`).

**Reason:** Cube updates are RL-style (store experiences) not differentiable.

**Gradients flow through:**
- Alpha gate (learned blending)
- Key projections (learned retrieval)
- Phase fusion (learned temporal encoding)

## Future Work

### 1. Full Wiring Telemetry

Extend `WiredMADModel` to:
- Thread `times` parameter through forward pass
- Aggregate telemetry from all blocks
- Return global HRM statistics

### 2. Liquid-mLSTM Hybrids

Explore:
- Liquid cells as post-processors within xLSTM blocks
- Mixed parallel/recurrent processing
- Per-head liquid dynamics

### 3. Multi-Modal Neuromodulation

Current: 5-HT (serotonin) only

Extend to:
- ACh (acetylcholine) for attention modulation
- DA (dopamine) for reward signals
- NE (norepinephrine) for arousal

### 4. Cube Merging and Distillation

For deployment:
- Merge cubes from similar contexts
- Distill cube predictions into feedforward network
- Knowledge graph extraction from cube contents

### 5. Benchmarking

Compare on:
- Long-context tasks (book comprehension)
- Few-shot learning (rapid adaptation)
- Memory-intensive tasks (fact recall)

## References

### HRM+ Research
- `docs/lnn_hrm_hybrid/` - 40+ documentation files
- `src/lnn_hrm/` - PyTorch reference implementation
- `examples/hrm_xlstm_demo.py` - MLX demos

### xLSTM-7B
- `docs/XLSTM_7B_ARCHITECTURE_MAP.md`
- `docs/CANONICAL_XLSTM_IMPLEMENTATION_NOTES.md`
- `xlstm_metal/blocks/mlstm_mlx/` - MLX implementation

### MAD Framework
- `docs/MAD_WIRING_INTEGRATION.md`
- `docs/MAD_COMPOSITION_PROPOSAL.md`
- `xlstm_metal/wiring/` - Core wiring system

## Contact and Contributions

This integration combines:
- xLSTM (Sepp Hochreiter et al., 2024)
- MAD framework (Hochreiter, LFM2)
- HRM+ research (this repository)
- MLX framework (Apple, 2024)

For questions or contributions, see repository README.
