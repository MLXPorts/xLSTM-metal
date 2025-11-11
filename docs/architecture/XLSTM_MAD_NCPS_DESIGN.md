# xLSTM-MAD-NCPS Architecture Design

**Date:** 2025-01-29  
**Status:** Design Proposal  
**Authors:** Architecture discussion synthesis

## Executive Summary

This document proposes a unified architecture for xLSTM that combines:

- **NCPS-style cell-level wiring** for fine-grained connectivity within blocks
- **MAD-style block composition** for heterogeneous layer stacking
- **LIV framework** (Linear Input-Varying systems) from STAR for flexible primitives
- **Backend-agnostic connectors** for cross-framework routing
- **HyperProfiles** for numerical equivalence across backends

## Architecture Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     xLSTM Architecture                          │
│                                                                 │
│  ┌──────────────┐   Connector    ┌──────────────┐             │
│  │              │  (Backend-agnostic routing)    │             │
│  │  mLSTM Block │◄────────────────► FFN Block    │             │
│  │              │    MLX ↔ PyTorch ↔ Numpy       │             │
│  └──────┬───────┘                  └──────┬──────┘             │
│         │                                  │                     │
│    MAD Wiring                         MAD Wiring                │
│    (Block-level)                      (Block-level)             │
│         │                                  │                     │
│  ┌──────▼──────────────────────────────────▼──────┐            │
│  │              LIV Layer (Cells)                  │            │
│  │                                                 │            │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │            │
│  │  │ Q-Cell  │  │ K-Cell  │  │ V-Cell  │       │            │
│  │  │ (LIV-1) │  │ (LIV-2) │  │ (LIV-3) │       │            │
│  │  └────┬────┘  └────┬────┘  └────┬────┘       │            │
│  │       │            │            │              │            │
│  │       └────────────┴────────────┘              │            │
│  │         Featurizer Weight Sharing              │            │
│  │                                                 │            │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │            │
│  │  │ i-Gate  │  │ f-Gate  │  │ o-Gate  │       │            │
│  │  │ (LIV-4) │  │ (LIV-5) │  │ (LIV-6) │       │            │
│  │  └────┬────┘  └────┬────┘  └────┬────┘       │            │
│  │       │            │            │              │            │
│  │       └────NCPS Wiring (sparsity masks)───┘   │            │
│  │         + polarity (erev) control              │            │
│  │                                                 │            │
│  │  ┌─────────────────────────────────────┐      │            │
│  │  │   Memory Update Cell (LIV-7)        │      │            │
│  │  │   - Matrix memory (mLSTM)           │      │            │
│  │  │   - Covariance update rule          │      │            │
│  │  └─────────────────────────────────────┘      │            │
│  │                                                 │            │
│  │  ┌─────────────────────────────────────┐      │            │
│  │  │   Norm Cell (LIV-8)                 │      │            │
│  │  │   - MultiHeadLayerNorm              │      │            │
│  │  └─────────────────────────────────────┘      │            │
│  │                                                 │            │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  HyperProfile: "mlstm_mlx_from_torch"                          │
│  - Initialization ranges                                       │
│  - Constraint ranges                                           │
│  - Solver config                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Current API vs. Proposed API

### Current Structure (xlstm_metal/blocks/mlx/mlstm/)

```python
# Current: Monolithic block
class mLSTMLayer(nn.Module):
    def __init__(self, config: mLSTMConfig):
        self.q = nn.Linear(...)
        self.k = nn.Linear(...)
        self.v = nn.Linear(...)
        self.igate_preact = nn.Linear(...)
        self.fgate_preact = nn.Linear(...)
        self.ogate_preact = nn.Linear(...)
        self.multihead_norm = MultiHeadLayerNorm(...)
        self.out_proj = nn.Linear(...)
    
    def __call__(self, x, state):
        # All computation inline
        # No internal wiring
        # No sparsity control
```

### Proposed Structure (Cell-Based with Wiring)

```python
# Proposed: Cell-based with NCPS wiring
from xlstm_metal.wiring import LIVWiring, CellSpec, BackendType
from xlstm_metal.profiles import HyperProfile, load_profile

# 1. Define cells as LIV specifications
cells = {
    'q_proj': CellSpec(
        name='q_proj',
        cell_type='linear_projection',
        backend=BackendType.MLX,
        params={'in_features': 4096, 'out_features': 2048},
        polarity=1.0  # Excitatory
    ),
    'k_proj': CellSpec(
        name='k_proj',
        cell_type='linear_projection',
        backend=BackendType.MLX,
        params={'in_features': 4096, 'out_features': 2048},
        polarity=1.0
    ),
    'v_proj': CellSpec(
        name='v_proj',
        cell_type='linear_projection',
        backend=BackendType.MLX,
        params={'in_features': 4096, 'out_features': 4096},
        polarity=1.0
    ),
    'igate': CellSpec(
        name='igate',
        cell_type='exponential_gate',
        backend=BackendType.MLX,
        params={'in_features': 4096, 'out_features': 8},
        polarity=1.0  # Excitatory (input gate)
    ),
    'fgate': CellSpec(
        name='fgate',
        cell_type='exponential_gate',
        backend=BackendType.MLX,
        params={'in_features': 4096, 'out_features': 8},
        polarity=-1.0  # Inhibitory (forget gate)
    ),
    'ogate': CellSpec(
        name='ogate',
        cell_type='sigmoid_gate',
        backend=BackendType.MLX,
        params={'in_features': 4096, 'out_features': 4096},
        polarity=1.0
    ),
    'memory': CellSpec(
        name='memory',
        cell_type='matrix_memory',
        backend=BackendType.MLX,
        params={'d': 512, 'num_heads': 8},
        polarity=1.0
    ),
    'norm': CellSpec(
        name='norm',
        cell_type='multihead_layernorm',
        backend=BackendType.MLX,
        params={'num_heads': 8, 'head_dim': 512},
        polarity=1.0
    )
}

# 2. Create wiring with sparse connectivity
wiring = LIVWiring(cells)

# Input feeds to Q/K/V projections (all in parallel)
wiring.add_connection('input', 'q_proj')
wiring.add_connection('input', 'k_proj')
wiring.add_connection('input', 'v_proj')

# Input also feeds gates (sparse - only some gates get direct input)
wiring.add_connection('input', 'igate', sparsity_mask=0.8)  # 80% sparse
wiring.add_connection('input', 'fgate', sparsity_mask=0.8)

# Q/K/V feed memory
wiring.add_connection('q_proj', 'memory')
wiring.add_connection('k_proj', 'memory')
wiring.add_connection('v_proj', 'memory')

# Gates modulate memory (polarity from CellSpec used)
wiring.add_connection('igate', 'memory')
wiring.add_connection('fgate', 'memory')

# Memory output goes through norm
wiring.add_connection('memory', 'norm')

# Output gate modulates normalized output
wiring.add_connection('ogate', 'norm')

# Share featurizers between Q/K projections (STAR-style)
wiring.add_featurizer_sharing(['q_proj', 'k_proj'], strategy='partial')

# Share feature groups (like GQA - grouped query attention)
wiring.add_feature_group_sharing(['k_proj', 'v_proj'], groups=['keys'])

# 3. Create wired mLSTM block
profile = load_profile('mlstm_mlx_from_torch')
mlstm_block = WiredmLSTMBlock(
    wiring=wiring,
    profile=profile,
    input_cell='input',
    output_cell='norm'
)
```

## Key Components

### 1. Cell Types (LIV Primitives)

Each cell is a Linear Input-Varying (LIV) operator from STAR framework:

```python
# xlstm_metal/cells/registry.py
CELL_REGISTRY = {
    # Projection cells
    'linear_projection': LinearProjectionCell,      # Dense W @ x
    'gated_projection': GatedProjectionCell,        # SwiGLU-style
    
    # Gate cells
    'exponential_gate': ExponentialGateCell,        # exp(Wx + b)
    'sigmoid_gate': SigmoidGateCell,                # σ(Wx + b)
    'soft_cap_gate': SoftCapGateCell,               # cap * tanh(x / cap)
    
    # Memory cells
    'matrix_memory': MatrixMemoryCell,              # mLSTM C ∈ R^(d×d)
    'scalar_memory': ScalarMemoryCell,              # sLSTM c ∈ R
    
    # Normalization cells
    'multihead_layernorm': MultiHeadLayerNormCell,
    'rmsnorm': RMSNormCell,
    
    # Recurrence cells
    'linear_recurrence': LinearRecurrenceCell,      # Semi-separable
    'gated_convolution': GatedConvolutionCell,      # Toeplitz
    
    # Attention cells
    'softmax_attention': SoftmaxAttentionCell,      # Dense attention
    'linear_attention': LinearAttentionCell,        # Low-rank
}
```

### 2. Wiring Abstraction

```python
# xlstm_metal/wiring/liv_wiring.py
class LIVWiring:
    """
    NCPS-style wiring for LIV cells within a block.
    
    Unlike MADWiring (block-to-block), this operates at the cell level
    and enforces sparse connectivity via sparsity masks.
    """
    
    def __init__(self, cell_specs: Dict[str, CellSpec]):
        self.cell_specs = cell_specs
        self.num_cells = len(cell_specs)
        
        # Adjacency matrix: [num_cells, num_cells]
        # A[i,j] = polarity * (1 - sparsity) if connected, else 0
        self.adjacency = mx.zeros((self.num_cells, self.num_cells))
        
        # Polarity (erev) from cell specs
        self.erev = mx.array([spec.polarity for spec in cell_specs.values()])
    
    def add_connection(
        self,
        from_cell: str,
        to_cell: str,
        polarity: Optional[float] = None,
        sparsity_mask: Optional[float] = 0.0  # 0.0 = fully connected
    ):
        """
        Add sparse connection between cells.
        
        Sparsity is enforced via element-wise multiplication during forward pass,
        just like NCPS ltc_cell.py lines 185-186 and 211-212.
        """
        from_idx = self.name_to_idx[from_cell]
        to_idx = self.name_to_idx[to_cell]
        
        if polarity is None:
            polarity = self.cell_specs[to_cell].polarity
        
        # Connectivity strength = polarity * (1 - sparsity)
        strength = polarity * (1.0 - sparsity_mask)
        self.adjacency[from_idx, to_idx] = strength
    
    def add_featurizer_sharing(
        self,
        cell_names: List[str],
        strategy: str = 'full'  # 'full', 'partial', 'none'
    ):
        """
        Share featurizer weights between cells (STAR-style).
        
        Example: Q and K projections share same featurizer but apply
        to different feature groups.
        """
        # Mark cells as sharing featurizer in genome
        for name in cell_names:
            self.featurizer_sharing[name] = cell_names[0]  # Share with first
            self.featurizer_sharing_strategy[name] = strategy
    
    def add_feature_group_sharing(
        self,
        cell_names: List[str],
        groups: List[str]  # ['keys', 'values']
    ):
        """
        Share feature groups directly (like GQA key/value sharing).
        """
        for name in cell_names:
            self.feature_group_sharing[name] = (cell_names[0], groups)
    
    def get_execution_stages(self) -> List[List[str]]:
        """
        Topological sort for parallel execution (like MADWiring).
        
        Cells in same stage can execute in parallel.
        """
        # Kahn's algorithm on adjacency matrix
        pass
```

### 3. Wired Block Implementation

```python
# xlstm_metal/blocks/mlx/mlstm/wired_block.py
class WiredmLSTMBlock(nn.Module):
    """
    mLSTM block with NCPS-style internal wiring.
    
    Analogous to WiredCfCCell in NCPS, but for xLSTM primitives.
    """
    
    def __init__(
        self,
        wiring: LIVWiring,
        profile: HyperProfile,
        input_cell: str,
        output_cell: str
    ):
        super().__init__()
        self._wiring = wiring
        self._profile = profile
        self._input_cell = input_cell
        self._output_cell = output_cell
        
        # Create cells from wiring specs
        self._cells: Dict[str, nn.Module] = {}
        for name, spec in wiring.cell_specs.items():
            cell_class = CELL_REGISTRY[spec.cell_type]
            
            # Apply HyperProfile for initialization
            init_config = profile.initializers.get(spec.cell_type, {})
            constraint_config = profile.constraints.get(spec.cell_type, {})
            
            self._cells[name] = cell_class(
                **spec.params,
                init_config=init_config,
                constraint_config=constraint_config
            )
            setattr(self, f'cell_{name}', self._cells[name])
        
        # Extract sparsity masks from wiring
        self._sparsity_masks = self._create_sparsity_masks()
    
    def _create_sparsity_masks(self) -> Dict[str, mx.array]:
        """
        Create sparsity masks for each cell connection.
        
        These are applied during forward pass to enforce wiring,
        just like NCPS multiplies activations by sparsity masks.
        """
        masks = {}
        for i, from_name in enumerate(self._wiring.cell_specs.keys()):
            for j, to_name in enumerate(self._wiring.cell_specs.keys()):
                if self._wiring.adjacency[i, j] != 0:
                    # Create mask from adjacency value
                    masks[(from_name, to_name)] = mx.abs(
                        self._wiring.adjacency[i, j]
                    )
        return masks
    
    def __call__(
        self,
        x: mx.array,
        state: Optional[Tuple] = None
    ) -> Tuple[mx.array, Optional[Tuple]]:
        """
        Forward pass with wiring enforcement.
        
        Execute cells in topological order, applying sparsity masks
        to connections (NCPS-style).
        """
        # Get execution stages for parallel execution
        stages = self._wiring.get_execution_stages()
        
        # Cell outputs
        cell_outputs = {self._input_cell: x}
        
        # Execute stage by stage
        for stage in stages:
            stage_outputs = {}
            
            # Cells in same stage can execute in parallel
            for cell_name in stage:
                if cell_name == self._input_cell:
                    continue
                
                # Gather inputs from connected cells
                cell_input = self._gather_cell_inputs(
                    cell_name,
                    cell_outputs
                )
                
                # Apply cell computation
                cell = self._cells[cell_name]
                stage_outputs[cell_name] = cell(cell_input, state)
            
            # Update outputs
            cell_outputs.update(stage_outputs)
        
        # Return output cell result
        return cell_outputs[self._output_cell]
    
    def _gather_cell_inputs(
        self,
        target_cell: str,
        cell_outputs: Dict[str, mx.array]
    ) -> mx.array:
        """
        Gather inputs for a cell from connected cells.
        
        Apply sparsity masks and polarity (NCPS-style).
        """
        target_idx = self._wiring.name_to_idx[target_cell]
        inputs = []
        
        for source_name, source_output in cell_outputs.items():
            source_idx = self._wiring.name_to_idx[source_name]
            
            if self._wiring.adjacency[source_idx, target_idx] != 0:
                # Apply sparsity mask (NCPS lines 185-186, 211-212)
                mask_key = (source_name, target_cell)
                if mask_key in self._sparsity_masks:
                    masked = mx.multiply(
                        source_output,
                        self._sparsity_masks[mask_key]
                    )
                    
                    # Apply polarity (erev)
                    polarity = self._wiring.adjacency[source_idx, target_idx]
                    polarity_sign = mx.sign(polarity)
                    masked = mx.multiply(masked, polarity_sign)
                    
                    inputs.append(masked)
        
        # Sum inputs (like NCPS accumulation)
        if not inputs:
            raise ValueError(f"No inputs for cell {target_cell}")
        
        return mx.sum(mx.stack(inputs), axis=0)
```

### 4. HyperProfile System

```python
# xlstm_metal/profiles/hyperprofile.py
@dataclass(frozen=True)
class HyperProfile:
    """
    Backend compensation for numerical equivalence.
    
    Ensures xLSTM trained in PyTorch can run identically in MLX
    by compensating for framework differences.
    """
    name: str
    description: str
    backend_source: str  # 'torch', 'mlx', 'jax'
    backend_target: str
    
    # Initialization ranges by cell type
    initializers: Dict[str, Dict[str, float]]
    
    # Constraint ranges for numerical stability
    constraints: Dict[str, Dict[str, float]]
    
    # Solver configuration (for ODE-based cells)
    solver_config: Dict[str, Any]
    
    # Framework-specific quirks
    extras: Dict[str, Any]

# Example profile JSON:
# xlstm_metal/profiles/mlstm_mlx_from_torch.json
{
  "name": "mlstm_mlx_from_torch",
  "description": "mLSTM running on MLX with PyTorch-equivalent numerics",
  "backend_source": "torch",
  "backend_target": "mlx",
  
  "initializers": {
    "exponential_gate": {
      "weight_min": 0.01,
      "weight_max": 1.0,
      "bias_min": 3.0,
      "bias_max": 6.0
    },
    "matrix_memory": {
      "init_scale": 0.02
    }
  },
  
  "constraints": {
    "exponential_gate": {
      "output_min": 1e-5,
      "output_max": 1e3
    },
    "matrix_memory": {
      "singular_value_min": 1e-6
    }
  },
  
  "solver_config": {
    "ode_unfolds": 6,
    "stabilizer_eps": 1e-8
  }
}
```

### 5. Connector System (Backend-Agnostic Routing)

```python
# xlstm_metal/connectors/base.py
class Connector:
    """
    Backend-agnostic connection between blocks.
    
    Inspired by LFM (Liquid Foundation Models) inter-block learning.
    Routes tensors between different backends (MLX ↔ PyTorch ↔ NumPy).
    """
    
    def __init__(
        self,
        source_backend: BackendType,
        target_backend: BackendType,
        learnable: bool = True  # Learn optimal routing like LFM
    ):
        self.source_backend = source_backend
        self.target_backend = target_backend
        self.learnable = learnable
        
        if learnable:
            # Learnable router (like LFM's inter-block layer)
            self.router = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
    
    def __call__(self, x: Any, route_metadata: Optional[Dict] = None) -> Any:
        """
        Route tensor between backends.
        
        If learnable, optimize routing based on downstream task performance.
        """
        # Convert source to intermediate (NumPy)
        if self.source_backend == BackendType.MLX:
            x_np = np.array(x)
        elif self.source_backend == BackendType.TORCH:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Apply learned routing if enabled
        if self.learnable and route_metadata:
            routing_weight = self.router(route_metadata['quality_metric'])
            x_np = x_np * float(routing_weight)
        
        # Convert intermediate to target
        if self.target_backend == BackendType.MLX:
            return mx.array(x_np)
        elif self.target_backend == BackendType.TORCH:
            return torch.from_numpy(x_np)
        else:
            return x_np

# Usage:
connector = Connector(BackendType.MLX, BackendType.TORCH, learnable=True)
torch_output = connector(mlx_output, {'quality_metric': perplexity})
```

## Integration with Existing Code

### Minimal Changes to Current API

```python
# OLD: xlstm_metal/blocks/mlx/mlstm/ffn_block.py
class mLSTMLayer(nn.Module):
    def __init__(self, config: mLSTMConfig):
        # ... monolithic implementation

# NEW: Backwards-compatible wrapper
class mLSTMLayer(nn.Module):
    def __init__(
        self,
        config: mLSTMConfig,
        use_wiring: bool = False,  # Opt-in to new system
        wiring: Optional[LIVWiring] = None,
        profile: Optional[HyperProfile] = None
    ):
        super().__init__()
        
        if use_wiring and wiring:
            # New wired implementation
            self._impl = WiredmLSTMBlock(wiring, profile, 'input', 'output')
        else:
            # Old monolithic implementation (unchanged)
            self._impl = _mLSTMLayerLegacy(config)
    
    def __call__(self, x, state):
        return self._impl(x, state)
```

## Migration Path

1. **Phase 1: Cell Registry** (Week 1-2)
    - Create cell type registry
    - Implement basic cells from existing code
    - No wiring yet, just modular cells

2. **Phase 2: Wiring Infrastructure** (Week 2-3)
    - Implement LIVWiring class
    - Add sparsity mask support
    - Topological execution ordering

3. **Phase 3: HyperProfile System** (Week 3-4)
    - Create profile loader
    - Add PyTorch→MLX profiles
    - Validate numerical equivalence

4. **Phase 4: Wired Blocks** (Week 4-5)
    - Implement WiredmLSTMBlock
    - Create example wirings
    - Benchmark against monolithic

5. **Phase 5: Connectors** (Week 5-6)
    - Implement basic Connector
    - Add learnable routing (LFM-style)
    - Cross-backend tests

6. **Phase 6: STAR Integration** (Week 6+)
    - Add genome encoding
    - Evolutionary optimization
    - Architecture search

## Benefits

1. **Fine-grained control**: Sparse connectivity within blocks
2. **Reusability**: Cells shared across mLSTM/sLSTM/FFN
3. **Flexibility**: Easy to add new cell types
4. **Backend portability**: HyperProfiles ensure numerical equivalence
5. **Cross-framework**: Connectors enable MLX↔PyTorch↔JAX
6. **Research-friendly**: Easy to experiment with architectures
7. **Backwards compatible**: Old API still works

## Open Questions

1. **Performance**: Does cell-level wiring add overhead?
2. **Memory**: How much extra memory for adjacency matrices?
3. **Parallelization**: Can we parallelize stages on GPU?
4. **Debugging**: How to visualize wiring graphs?
5. **Serialization**: How to save/load wired models?

## Next Steps

1. Review this design with team
2. Create prototype of cell registry
3. Implement simple wiring example
4. Benchmark vs. current implementation
5. Iterate based on findings
