# xLSTM Component Architecture Analysis

**Date:** 2025-01-29  
**Context:** Understanding the block/component structure for NCPS-style wiring

## Current Understanding: Three Layers of Abstraction

### Layer 1: Metal Kernels (Lowest Level - GPU Code)

**Registry:** `xlstm_metal/blocks/mlx/mlstm/metal_kernels/kernel_registry.py`

**Pattern:** Singleton with lazy JIT compilation
```python
# At module import time (mlstm_chunkwise_recurrent_fw_C.py line 349):
register_kernel('fw_recurrent', _compile_recurrent_kernel)

# On first use (lazy compilation):
kernel = get_kernel('fw_recurrent')  # Compiles once, caches forever
outputs = kernel(inputs=[...], grid=..., threadgroup=...)
```

**Kernels:**
- `fw_recurrent` - Single-step recurrent forward
- `fw_parallel` - Chunkwise parallel forward
- `bw_recurrent` - Backward pass recurrent
- `bw_parallel_dQ`, `bw_parallel_dK`, `bw_parallel_dV` - Backward gradients

**Critical:** These compile ONCE on first access and stay in memory as global objects.

### Layer 2: Kernel Strategies (Python-Level Dispatch)

**Location:** `xlstm_metal/blocks/mlx/mlstm/kernels/`

**Current implementation:**
- ❌ No registry pattern
- ❌ Direct imports: `from .kernel import mlstm_sequential`
- ❌ Hardcoded in block.py

**Functions:**
- `mlstm_recurrent_step()` - Single step inference (calls Metal kernel)
- `mlstm_sequential()` - Loop over steps (calls recurrent_step)
- `mlstm_chunkwise()` - Parallel chunks (calls Metal parallel kernel)

**Should be:** NCPS-style string → Callable registry (see proposal below)

### Layer 3: Blocks (High-Level Composition)

**From config.json and model weights:**

```
xLSTM-7B Architecture:
├── embedding (vocab_size=50304, d_model=4096)
├── backbone
│   ├── blocks.0
│   │   ├── norm_mlstm (RMSNorm, eps=1e-6)
│   │   ├── mlstm_layer (mLSTMLayer)
│   │   │   ├── q.weight [2048, 4096]
│   │   │   ├── k.weight [2048, 4096]
│   │   │   ├── v.weight [4096, 4096]
│   │   │   ├── igate_preact.weight [8, 4096]
│   │   │   ├── igate_preact.bias [8]
│   │   │   ├── fgate_preact.weight [8, 4096]
│   │   │   ├── fgate_preact.bias [8]
│   │   │   ├── ogate_preact.weight [4096, 4096]
│   │   │   ├── multihead_norm.weight [8, 512]
│   │   │   └── out_proj.weight [4096, 4096]
│   │   ├── norm_ffn (RMSNorm, eps=1e-6)
│   │   └── ffn (FeedForward)
│   │       ├── proj_up_gate.weight [10944, 4096]
│   │       ├── proj_up.weight [10944, 4096]
│   │       └── proj_down.weight [4096, 10944]
│   ├── blocks.1
│   │   └── (same structure)
│   ├── ...
│   └── blocks.31
│       └── (same structure)
└── out_norm (RMSNorm)
```

**Key Observations:**

1. **32 identical blocks** (config: `num_blocks: 32`)
2. **4 components per block:**
   - `norm_mlstm` - Pre-normalization before mLSTM
   - `mlstm_layer` - The mLSTM computation
   - `norm_ffn` - Pre-normalization before FFN
   - `ffn` - FeedForward network (SwiGLU variant)

3. **No sLSTM in inference!** Only mLSTM blocks (matches our earlier finding)

4. **Config specifies kernel strategies:**
   ```json
   "chunkwise_kernel": "chunkwise--triton_xl_chunk",
   "sequence_kernel": "native_sequence__triton",
   "step_kernel": "triton",
   "mode": "inference"
   ```

## Component Breakdown

### mLSTM Layer Components (Inside mlstm_layer)

From weight structure, the mLSTM layer contains:

```python
# Projections (components)
- Q projection: Linear(4096 → 2048)
- K projection: Linear(4096 → 2048)  
- V projection: Linear(4096 → 4096)

# Gates (components)
- Input gate: Linear(4096 → 8, bias=True)  # Per-head!
- Forget gate: Linear(4096 → 8, bias=True)  # Per-head!
- Output gate: Linear(4096 → 4096)

# Normalization (component)
- MultiHeadLayerNorm: [8 heads, 512 head_dim]

# Output (component)
- Output projection: Linear(4096 → 4096)

# Kernel (not a weight, but a computation component)
- mLSTM kernel: Strategy-selected (recurrent/sequential/parallel)
```

### FFN Components

```python
# FeedForward is SwiGLU variant:
- proj_up_gate: Linear(4096 → 10944)  # Gate path
- proj_up: Linear(4096 → 10944)       # Value path
- proj_down: Linear(10944 → 4096)     # Down projection
- activation: SiLU()

# Computation: down_proj(silu(up_gate) * up_proj)
```

## Proposed Component-Based Refactoring

### Structure

```
xlstm_metal/
├── blocks/                    # High-level blocks
│   └── mlx/
│       ├── mlstm/
│       │   ├── block.py              # mLSTMBlock (composition)
│       │   ├── components/           # ← NEW: Reusable components
│       │   │   ├── __init__.py
│       │   │   ├── projections.py    # Q/K/V projections
│       │   │   ├── gates.py          # Input/Forget/Output gates
│       │   │   ├── norms.py          # RMSNorm, MultiHeadLayerNorm
│       │   │   └── registry.py       # Component registry
│       │   ├── kernels/              # Kernel strategies
│       │   │   ├── __init__.py
│       │   │   ├── recurrent.py      # Single-step strategy
│       │   │   ├── sequential.py     # Loop strategy
│       │   │   ├── parallel.py       # Chunkwise parallel strategy
│       │   │   └── strategy_registry.py  # ← NEW: NCPS-style registry
│       │   └── metal_kernels/        # JIT-compiled GPU code
│       │       ├── kernel_registry.py    # Singleton (already exists)
│       │       ├── fw_kernel_recurrent.py
│       │       └── fw_kernel_parallel.py
│       └── ffn/
│           └── block.py              # FeedForward
└── profiles/                  # ← NEW: HyperProfiles
    ├── mlstm_mlx_from_torch.json
    └── mlstm_inference.json
```

### Component Registry Pattern (NCPS-Style)

**Example: Gate Components**

```python
# xlstm_metal/blocks/mlx/mlstm/components/gates.py
from typing import Callable, Optional
import mlx.core as mx
import mlx.nn as nn

class GateRegistry:
    """Registry for gate types with NCPS pattern"""
    
    _gate_types = {}
    
    @classmethod
    def register(cls, name: str, gate_class):
        """Register a gate type"""
        cls._gate_types[name] = gate_class
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """Get gate instance by name"""
        if name not in cls._gate_types:
            raise ValueError(f"Unknown gate type: {name}")
        return cls._gate_types[name](**kwargs)

# Register gate types
@GateRegistry.register("input_gate")
class InputGate(nn.Module):
    def __init__(self, input_dim, num_heads, soft_cap=15.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_heads, bias=True)
        self.soft_cap = soft_cap
    
    def __call__(self, x):
        i_preact = self.proj(x)
        if self.soft_cap:
            i_preact = self.soft_cap * mx.tanh(i_preact / self.soft_cap)
        return i_preact

@GateRegistry.register("forget_gate")
class ForgetGate(nn.Module):
    # ... similar
```

**Usage in mLSTMLayer:**

```python
from .components import GateRegistry, ProjectionRegistry, NormRegistry
from .kernels import StrategyRegistry

class mLSTMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Components from registry (config-driven)
        self.q_proj = ProjectionRegistry.get("multihead", 
            input_dim=config.embedding_dim,
            num_heads=config.num_heads,
            head_dim=config.qk_head_dim
        )
        
        self.igate = GateRegistry.get("input_gate",
            input_dim=config.embedding_dim,
            num_heads=config.num_heads,
            soft_cap=config.gate_soft_cap
        )
        
        # Kernel strategy from registry
        self.kernel = StrategyRegistry.get_strategy("auto",
            chunk_size=config.chunk_size,
            eps=config.eps
        )
```

## Initialization Order (Critical for Metal Kernels)

Based on the Metal kernel compilation pattern, the correct initialization order is:

1. **Module import time:**
   ```python
   # mlstm_chunkwise_recurrent_fw_C.py (line 349)
   register_kernel('fw_recurrent', _compile_recurrent_kernel)
   ```
   This registers the COMPILER FUNCTION (not the compiled kernel).

2. **First use (lazy compilation):**
   ```python
   # When mlstm_sequential() first calls:
   kernel = get_kernel('fw_recurrent')  # Compiles here, caches result
   ```

3. **Subsequent uses:**
   ```python
   kernel = get_kernel('fw_recurrent')  # Returns cached kernel
   ```

**This means:**
- ✅ Kernels compile ONCE per process lifetime
- ✅ Compilation is lazy (on first access)
- ✅ Kernels are global objects (singleton registry)
- ❌ Kernels are NOT pickled/saved (they're JIT-compiled code)
- ❌ Kernels are NOT passed through function arguments

## Config-Driven Kernel Selection

The config.json specifies kernel strategies:

```json
{
  "chunkwise_kernel": "chunkwise--triton_xl_chunk",
  "sequence_kernel": "native_sequence__triton",
  "step_kernel": "triton",
  "mode": "inference"
}
```

**Translation to our system:**
- `chunkwise_kernel` → Use parallel strategy (`mlstm_chunkwise`)
- `sequence_kernel` → Use sequential strategy (`mlstm_sequential`)
- `step_kernel` → Use recurrent strategy (`mlstm_recurrent_step`)
- `mode: inference` → No backward kernels needed

**Proposed StrategyConfig:**

```python
@dataclass
class StrategyConfig:
    """Kernel strategy configuration from model config"""
    
    # From config.json
    mode: str = "inference"  # "inference" or "train"
    chunkwise_kernel: str = "parallel"
    sequence_kernel: str = "sequential"
    step_kernel: str = "recurrent"
    chunk_size: int = 64
    
    # Numerical stability
    eps: float = 1e-6
    gate_soft_cap: float = 15.0
    inference_state_dtype: str = "float32"
    
    @classmethod
    def from_model_config(cls, config_dict):
        """Create from loaded config.json"""
        return cls(
            mode=config_dict.get("mode", "inference"),
            chunk_size=config_dict.get("chunk_size", 64),
            eps=config_dict.get("eps", 1e-6),
            gate_soft_cap=config_dict.get("gate_soft_cap", 15.0),
            inference_state_dtype=config_dict.get("inference_state_dtype", "float32")
        )
```

## HyperProfile Integration

HyperProfiles should compensate for backend differences:

```json
// xlstm_metal/profiles/mlstm_mlx_from_torch.json
{
  "name": "mlstm_mlx_from_torch",
  "description": "mLSTM on MLX with PyTorch-equivalent numerics",
  "backend_source": "torch",
  "backend_target": "mlx",
  
  "kernel_strategies": {
    "recurrent": "mlstm_recurrent_step",
    "sequential": "mlstm_sequential",
    "parallel": "mlstm_chunkwise"
  },
  
  "dtype_management": {
    "state_dtype": "float32",
    "computation_dtype": "float32",
    "cast_for_matmul": true
  },
  
  "numerical_stability": {
    "eps": 1e-6,
    "gate_soft_cap": 15.0,
    "use_logsigmoid": true,
    "denominator_floor": 1e-30
  },
  
  "gate_initialization": {
    "igate_bias_init_range": -10.0,
    "fgate_bias_init_range": 0.0
  }
}
```

## Key Takeaways

1. **Three-layer architecture:** Metal kernels → Kernel strategies → Blocks
2. **Metal kernels compile once** - lazy compilation on first use
3. **32 identical blocks** - mLSTM + FFN pattern repeated
4. **Config drives behavior** - kernel strategies, hyperparams from config.json
5. **No sLSTM in inference** - only used during training
6. **Component-based refactoring** - NCPS pattern with registries
7. **HyperProfiles** - Backend compensation for numerical equivalence

## Next Steps

1. ✅ Understand block structure (DONE)
2. ✅ Understand Metal kernel compilation (DONE)
3. ⏭️ **Fix dtype management issue** (HIGH PRIORITY)
4. ⏭️ Create StrategyRegistry for kernel selection
5. ⏭️ Extract components (gates, projections, norms)
6. ⏭️ Create HyperProfiles for MLX-from-PyTorch
7. ⏭️ Add NCPS-style wiring support
