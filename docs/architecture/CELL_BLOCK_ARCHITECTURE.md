# xLSTM Cell/Block Architecture with NCPS Wiring

## Key Insight from NCPS and xLSTM-1B

Based on studying NCPS MLX and the xLSTM architecture differences:

### Architecture Hierarchy

```
Model (WiredMADModel)
  └── Uses: Top-level Wiring (defines block connectivity)
  └── Contains: Blocks (xLSTMBlock, sLSTMBlock, etc.)
       └── Each Block IS a Cell
       └── Uses: Internal Wiring (for internal connectomes)
       └── May use: Weight sharing via wiring patterns
       └── Contains: Sub-components (mLSTM, FFN, Conv1d, etc.)
```

### The Critical Distinction

**From NCPS:**
- `WiredCfCCell` takes a `Wiring` object
- Creates multiple `CfCCell` instances (one per layer in wiring)
- Applies connectivity pattern via sparsity masks
- Each cell has its own parameters

**For xLSTM:**
- Each **Block** (mLSTM, sLSTM) is actually a **Cell**
- Blocks can have internal wiring (connectomes within the block)
- Blocks may share weights via wiring patterns
- Different model sizes (1B vs 7B) use different block types

### xLSTM-7B vs xLSTM-1B Differences

**xLSTM-7B:**
- Simple sequential blocks
- All mLSTM blocks
- No Conv1d
- No sLSTM
- Sequential architecture

**xLSTM-1B:**
- Mixed block types (mLSTM + sLSTM + Conv1d)
- More complex connectivity
- Recurrence patterns
- Weight sharing opportunities
- Each block type is a cell

## Proposed Architecture

### 1. Base Cell Interface

```python
class xLSTMCell(nn.Module):
    """
    Base class for xLSTM cells.
    
    A cell is a computational unit that:
    - Has internal wiring (connectome)
    - May use sparsity patterns
    - Has its own state
    - Can be composed into larger circuits
    """
    
    def __init__(self, config, wiring: Optional[Wiring] = None):
        super().__init__()
        self.config = config
        self._wiring = wiring
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using wiring patterns."""
        raise NotImplementedError
    
    def __call__(self, x, state):
        """Forward pass through cell."""
        raise NotImplementedError
```

### 2. Concrete Cell Implementations

```python
class mLSTMCell(xLSTMCell):
    """
    mLSTM cell with matrix memory.
    
    Internal components:
    - Query/Key/Value projections (may use sparsity)
    - Input/Forget/Output gates
    - Multi-head layer norm
    - Output projection
    
    Can use internal wiring for:
    - Head connectivity patterns
    - Sparse projections
    - Weight sharing across heads
    """
    
    def _init_parameters(self):
        # Use wiring if provided for sparsity patterns
        if self._wiring:
            # Apply sparsity masks to projections
            self._apply_wiring_patterns()
        
        # Standard mLSTM components
        self.q = nn.Linear(...)
        self.k = nn.Linear(...)
        # ... etc


class sLSTMCell(xLSTMCell):
    """
    sLSTM cell with scalar recurrence.
    
    Simpler than mLSTM:
    - Block-diagonal recurrence
    - Scalar gates
    - More efficient for longer sequences
    """


class Conv1dCell(xLSTMCell):
    """
    Conv1d cell for local feature mixing.
    
    Used in xLSTM-1B for:
    - Local context aggregation
    - Feature mixing between blocks
    - Efficient local processing
    """
```

### 3. Block = Cell with Wrapper

```python
class xLSTMBlock(nn.Module):
    """
    xLSTM block wraps a cell with:
    - Pre-normalization
    - Residual connections
    - FFN (optional)
    
    This is the unit that gets wired together in the top-level circuit.
    """
    
    def __init__(self, config, cell_type="mlstm", cell_wiring=None):
        super().__init__()
        
        # Pre-norms
        self.xlstm_norm = RMSNorm(...)
        self.ffn_norm = RMSNorm(...)
        
        # Cell (the actual computation)
        if cell_type == "mlstm":
            self.cell = mLSTMCell(config, cell_wiring)
        elif cell_type == "slstm":
            self.cell = sLSTMCell(config, cell_wiring)
        elif cell_type == "conv1d":
            self.cell = Conv1dCell(config, cell_wiring)
        
        # FFN
        self.ffn = GatedFFN(...)
    
    def __call__(self, x, state):
        # mLSTM path with residual
        h = self.xlstm_norm(x)
        h, new_state = self.cell(h, state)
        x = x + h  # residual
        
        # FFN path with residual
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h  # residual
        
        return x, new_state
```

### 4. Top-Level Wiring

```python
class xLSTMModel(nn.Module):
    """
    Complete xLSTM model using NCPS wiring.
    
    Wiring defines:
    - Which blocks connect to which
    - Excitatory/inhibitory connections
    - Sparse connectivity patterns
    - Block ordering and parallelism
    """
    
    def __init__(self, wiring: Wiring, block_configs: Dict):
        super().__init__()
        self._wiring = wiring
        self._blocks = []
        
        # Create blocks based on wiring topology
        for layer_idx in range(wiring.num_layers):
            neurons = wiring.get_neurons_of_layer(layer_idx)
            
            for neuron_id in neurons:
                neuron_type = wiring.get_type_of_neuron(neuron_id)
                
                # Get block type from configuration
                block_type = block_configs[neuron_type]["type"]
                cell_config = block_configs[neuron_type]["config"]
                
                # Get sparsity pattern for this neuron
                cell_wiring = self._extract_cell_wiring(wiring, neuron_id)
                
                # Create block with appropriate cell
                block = xLSTMBlock(
                    config=cell_config,
                    cell_type=block_type,
                    cell_wiring=cell_wiring
                )
                
                self._blocks.append(block)
                setattr(self, f"block_{neuron_id}", block)
```

### 5. Weight Sharing via Wiring

```python
class WeightSharedWiring(Wiring):
    """
    Wiring that defines weight sharing patterns.
    
    Example: Multiple heads sharing the same projection weights.
    """
    
    def __init__(self, units, share_groups):
        super().__init__(units)
        self.share_groups = share_groups
    
    def get_shared_params(self, neuron_id):
        """Return which parameters this neuron shares."""
        for group in self.share_groups:
            if neuron_id in group:
                return group[0]  # Share with first in group
        return neuron_id  # No sharing
```

## Implementation Strategy

### Phase 1: Refactor Current Code
1. Make `xLSTMBlock` wrap a `mLSTMCell`
2. Extract cell logic from block logic
3. Add support for cell-level wiring

### Phase 2: Add Cell Types
1. Implement `sLSTMCell`
2. Implement `Conv1dCell`
3. Add cell factory to `xLSTMBlock`

### Phase 3: Advanced Wiring
1. Implement weight sharing via wiring
2. Add sparse connectivity patterns
3. Support heterogeneous block types

### Phase 4: Model Variants
1. xLSTM-7B: Simple sequential (current)
2. xLSTM-1B: Mixed cells with complex wiring
3. Custom: User-defined wiring patterns

## Key Benefits

1. **Modularity**: Cells are composable units
2. **Flexibility**: Easy to add new cell types
3. **Efficiency**: Weight sharing reduces parameters
4. **Research**: Easy to experiment with connectivity
5. **Compatibility**: Matches NCPS architecture patterns

## References

- NCPS MLX: `WiredCfCCell` pattern
- xLSTM Paper: Block specifications
- LTC Cell: Internal mixing patterns
- CfC Cell: Sparsity mask application

