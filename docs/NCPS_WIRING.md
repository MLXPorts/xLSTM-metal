# NCPS Auto-Wiring System

**Author:** Sydney Renee  
**Inspiration:** [ncps-mlx](https://github.com/MLXPorts/ncps-mlx) Neural Circuit Policy wiring patterns  
**Status:** Production (January 2025)

This document describes the NCPS-inspired wiring system that enables config-driven model composition.

## What Problem Does This Solve?

**Traditional approach** (hardcoded):
```python
class xLSTM7B(nn.Module):
    def __init__(self):
        self.blocks = nn.ModuleList([
            mLSTMBlock(...),  # Block 0
            mLSTMBlock(...),  # Block 1
            # ... hardcode 32 blocks
            mLSTMBlock(...),  # Block 31
        ])
```

**Problems:**
- Model size hardcoded (can't handle 1B, 13B, etc. without code changes)
- Block types hardcoded (what if checkpoint has attention layers?)
- Structure discovery manual (inspect checkpoint, write code, test)
- Not forward-compatible (new checkpoint formats need new code)

**NCPS solution** (introspective):
```python
wiring = AutoWiring.from_safetensors("xlstm_7b_model")
# Automatically detects:
# - 32 blocks
# - All mLSTM type
# - Sequential connectivity
# - Correct dimensions from weights

model = WiredxLSTM(config, wiring)  # Just works
```

## Core Concepts

### 1. Wiring as Structure Definition

A **wiring** defines:
- How many blocks (neurons in NCPS terminology)
- What types of blocks (mLSTM, sLSTM, attention, etc.)
- How they connect (sequential, parallel, skip connections)
- No actual computation - just structure

Think of it as a **blueprint**, not the building.

### 2. Introspection Over Hardcoding

Instead of telling the model its structure, **ask the checkpoint**:

```python
def discover_structure(safetensors_path):
    # Read index file
    index = json.load(open(f"{path}/model.safetensors.index.json"))
    
    # Find all block indices
    block_keys = [k for k in index["weight_map"] if "blocks." in k]
    block_indices = set()
    for key in block_keys:
        match = re.search(r"blocks\.(\d+)\.", key)
        if match:
            block_indices.add(int(match.group(1)))
    
    num_blocks = max(block_indices) + 1
    
    # Detect block types from weight patterns
    block_types = []
    for i in range(num_blocks):
        if any(f"blocks.{i}.mlstm" in k for k in block_keys):
            block_types.append("mlstm")
        elif any(f"blocks.{i}.slstm" in k for k in block_keys):
            block_types.append("slstm")
        else:
            block_types.append("unknown")
    
    return {"num_blocks": num_blocks, "block_types": block_types}
```

This works for **any** xLSTM checkpoint without code changes.

### 3. Factory Pattern for Block Creation

Wiring provides factory methods, not instances:

```python
class AutoWiring(Wiring):
    def create_block_cell(self, block_idx, config, **overrides):
        """Create a block cell at the given index."""
        block_type = self.structure['block_types'][block_idx]
        
        if block_type == "mlstm":
            return mLSTMBlock.from_config(config, **overrides)
        elif block_type == "slstm":
            return sLSTMBlock.from_config(config, **overrides)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
```

**Why factories?**
- Allows runtime override of parameters (dtype, chunk_size, etc.)
- Enables A/B testing (try different implementations)
- Supports mixed precision (different dtypes per block)

## NCPS Inspiration

Neural Circuit Policies (NCP) use **sparse wiring patterns** to define recurrent networks. Key ideas we adapted:

### 1. Explicit Connectivity

NCPS defines connections as **adjacency matrix**:
```
     0  1  2  3  (destination neurons)
  0 [0  1  0  0]
  1 [0  0  1  0]
  2 [0  0  0  1]
  3 [0  0  0  0]
(source neurons)
```

For xLSTM, this is simple sequential:
```
block_0 → block_1 → block_2 → ... → block_31
```

But the framework supports **any** pattern (skip connections, parallel paths, etc.).

### 2. Sensory/Inter/Motor Layers

NCPS organizes neurons into functional layers:
- **Sensory**: Input processing
- **Inter**: Hidden recurrent dynamics
- **Motor**: Output generation

We map this to xLSTM:
- **Sensory**: Embedding layer
- **Inter**: mLSTM/sLSTM blocks
- **Motor**: Output norm + LM head

### 3. Polarity (Excitatory/Inhibitory)

NCPS connections have polarity (sign):
- +1 = excitatory (add signal)
- -1 = inhibitory (subtract signal)

We use this for:
- Residual connections (+1)
- Optional gating (-1 for forget gates)

### 4. Sparsity

NCPS allows sparse connections (not all neurons connect). Benefits:
- Reduced computation
- Better generalization
- Architectural inductive biases

Currently xLSTM is fully sequential (dense path), but framework supports sparse patterns for future work.

## Implementation

### Core Classes

**File:** `xlstm_metal/mlx_jit/wiring/wirings.py`

```python
class Wiring:
    """Base class for wiring patterns."""
    
    def __init__(self, units: int):
        self.units = units
        self.adjacency_matrix = mx.zeros((units, units))
        self.sensory_adjacency_matrix = mx.zeros((units, num_inputs))
    
    def add_synapse(self, src: int, dest: int, polarity: int = 1):
        """Add connection between neurons."""
        self.adjacency_matrix[src, dest] = polarity
    
    def get_execution_order(self) -> List[List[int]]:
        """Compute execution order (topological sort)."""
        # Returns list of "stages" where neurons in same stage
        # can execute in parallel
        ...
```

**File:** `xlstm_metal/mlx_jit/wiring/auto_wiring.py`

```python
class AutoWiring(Wiring):
    """Auto-discovery wiring from safetensors."""
    
    @classmethod
    def from_safetensors(cls, model_path: str) -> "AutoWiring":
        """Introspect model structure from checkpoint."""
        structure = discover_structure(model_path)
        
        wiring = cls(units=structure['num_blocks'])
        
        # Sequential connections
        for i in range(structure['num_blocks'] - 1):
            wiring.add_synapse(i, i+1, polarity=1)
        
        wiring.structure = structure
        return wiring
    
    def create_block_cell(self, block_idx, config, **overrides):
        """Factory method for block creation."""
        # Dispatches to correct block type
        ...
```

### Usage in WiredxLSTM

**File:** `xlstm_metal/mlx_jit/models/wired_xlstm.py`

```python
class WiredxLSTM(mx.nn.Module):
    def __init__(self, config: Dict, wiring: AutoWiring):
        super().__init__()
        
        # Embedding (sensory layer)
        self.embedding = mx.nn.Embedding(config['vocab_size'], config['embedding_dim'])
        
        # Blocks (inter layer) - created via wiring
        self.blocks = []
        for i in range(wiring.structure['num_blocks']):
            block = wiring.create_block_cell(
                block_idx=i,
                config=config,
                compute_dtype=self.compute_dtype,
                state_dtype=self.state_dtype
            )
            self.blocks.append(block)
        
        # Output (motor layer)
        self.out_norm = mx.nn.RMSNorm(config['embedding_dim'])
        self.lm_head = mx.nn.Linear(config['embedding_dim'], config['vocab_size'])
    
    def __call__(self, input_ids, state=None):
        x = self.embedding(input_ids)
        
        # Execute blocks in wiring order
        new_state = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state is not None else None
            x, s = block(x, block_state)
            new_state.append(s)
        
        x = self.out_norm(x)
        logits = self.lm_head(x)
        
        return logits, new_state
```

## Benefits

### 1. Model-Agnostic Code

**Same code handles all xLSTM variants:**
- xLSTM-1B (16 blocks)
- xLSTM-7B (32 blocks)
- xLSTM-13B (40 blocks, hypothetical)
- Future architectures with mixed block types

**No if/else chains for model size.** Structure discovery handles it.

### 2. Forward Compatibility

When NX-AI releases new checkpoint formats:
```python
# Old format
"backbone.blocks.0.mlstm.proj_q.weight"

# New format (hypothetical)
"transformer.layers.0.mlstm_cell.query_proj.weight"
```

**Only change:** Update regex patterns in `discover_structure()`. Model code unchanged.

### 3. Architectural Experimentation

Want to try different connection patterns?

```python
# Sequential (current)
wiring = AutoWiring.from_safetensors("model")

# Skip connections (experimental)
wiring = AutoWiring.from_safetensors("model")
for i in range(0, wiring.units-2, 2):
    wiring.add_synapse(i, i+2, polarity=1)  # Skip every other block

# Parallel heads (future)
wiring = ParallelHeadsWiring(num_heads=4, head_size=8)
```

**Change wiring, not model code.**

### 4. Debugging and Visualization

Wiring is **introspectable**:

```python
wiring = AutoWiring.from_safetensors("xlstm_7b_model")

print(f"Blocks: {wiring.structure['num_blocks']}")
print(f"Types: {wiring.structure['block_types']}")

# Visualize connectivity
wiring.visualize()  # Prints ASCII art of connections
```

See `xlstm_metal/mlx_jit/wiring/wirings.py` `visualize()` method.

## Comparison to MAD Wiring

Earlier versions used "MAD (Modular Architecture Design) wiring" with explicit stages:

```python
# MAD approach (archived)
stage_0 = [mLSTMBlock(), mLSTMBlock(), mLSTMBlock(), mLSTMBlock()]
stage_1 = [mLSTMBlock(), mLSTMBlock(), mLSTMBlock(), mLSTMBlock()]
# ... 7 stages total

wiring = MADWiring(stages=[stage_0, stage_1, ...])
```

**Why we switched to NCPS:**

| Aspect | MAD Wiring | NCPS AutoWiring |
|--------|------------|------------------|
| Structure | Hardcoded stages | Introspected |
| Parallelism | Stage-level | Optional (currently sequential) |
| Flexibility | Fixed topology | Arbitrary graphs |
| Complexity | High (sync barriers) | Low (simple factories) |
| Debugging | Hard (race conditions) | Easy (deterministic) |
| Parity | Different from canonical | Matches canonical |

**NCPS is simpler and more maintainable** for current needs. MAD may return if we need stage-level parallelism.

See `docs_archive/components/mad/` for MAD design docs.

## Future Directions

### 1. Parallel Execution

Current implementation is sequential:
```python
for block in blocks:
    x, state = block(x, state)
```

NCPS wiring supports parallel execution within stages:
```python
stages = wiring.get_execution_order()

for stage in stages:
    # All blocks in stage can run in parallel
    outputs = []
    for block_idx in stage:
        out, state = blocks[block_idx](x, states[block_idx])
        outputs.append(out)
    
    # Combine outputs (sum, concat, etc.)
    x = combine(outputs)
```

### 2. Hierarchical Wiring

For long contexts, could implement hierarchical structure:

```
Short-term blocks (recent 1K tokens)
  ↓
Medium-term blocks (summaries 8K tokens)
  ↓  
Long-term blocks (compressed >32K tokens)
```

NCPS naturally represents this as multi-layer adjacency.

### 3. Dynamic Routing

With sparse wiring, could implement dynamic routing:
```python
# Route based on content
if is_code(input):
    active_blocks = code_expert_blocks
else:
    active_blocks = text_expert_blocks

x = wiring.execute(x, active_blocks=active_blocks)
```

Mixture-of-Experts without MoE routing layer.

### 4. Architecture Search

Generate random wiring patterns, evaluate, evolve:
```python
population = [RandomWiring(units=32) for _ in range(100)]

for generation in range(1000):
    # Evaluate each wiring
    scores = [evaluate(wiring) for wiring in population]
    
    # Select, mutate, crossover
    population = evolve(population, scores)

best_wiring = max(population, key=evaluate)
```

NCPS makes architecture search **first-class**.

## Comparison to Other Frameworks

### PyTorch (Sequential/ModuleList)

```python
# PyTorch
self.layers = nn.Sequential(
    Layer1(),
    Layer2(),
    Layer3(),
)

# Or
self.layers = nn.ModuleList([Layer1(), Layer2(), Layer3()])
```

**Limitations:**
- Structure hardcoded
- Can't introspect before instantiation
- No connection graph (just list)
- Sequential or nothing

### JAX (Pytrees)

```python
# JAX
model = {
    'embed': embed_params,
    'blocks': [block1_params, block2_params, ...],
    'head': head_params
}
```

**Limitations:**
- Pure data (no structure information)
- Must know structure to traverse
- No factory methods

### Keras Functional API

```python
# Keras
input = Input(shape=(seq_len,))
x = Embedding(...)(input)
x = Layer1()(x)
x = Layer2()(x)
output = Layer3()(x)
model = Model(inputs=input, outputs=output)
```

**Better:** Explicit graph construction

**Limitations:**
- Still manual specification
- Hard to generate programmatically
- No introspection from weights

### NCPS Advantage

**Structure is data**, not code. Can be:
- Generated from checkpoint
- Serialized/deserialized
- Visualized
- Evolved
- Analyzed mathematically

## Related Work

**Original NCPS:**
- Lechner et al., "Neural Circuit Policies Enable Fast Task Inference," NeurIPS 2020
- GitHub: https://github.com/mlech26l/ncps

**Our MLX port:**
- [ncps-mlx](https://github.com/MLXPorts/ncps-mlx) - Full NCPS implementation for MLX
- Includes wiring patterns: FullyConnected, Random, NCP, etc.

**xLSTM-Metal adaptation:**
- Simplified to AutoWiring (introspection-focused)
- Removed LTC neuron dynamics (not needed for xLSTM)
- Added safetensors introspection
- Optimized for sequential execution

## Summary

NCPS auto-wiring provides:
1. **Config-driven architecture** - No hardcoded model sizes
2. **Forward compatibility** - Works with new checkpoint formats
3. **Architectural flexibility** - Easy to experiment with connections
4. **Clean abstraction** - Structure separate from computation
5. **Introspection** - Understand model before loading

**It's what makes this port work with any xLSTM variant without code changes.**

---

For implementation details, see:
- `xlstm_metal/mlx_jit/wiring/wirings.py` (base classes)
- `xlstm_metal/mlx_jit/wiring/auto_wiring.py` (auto-discovery)
- `xlstm_metal/mlx_jit/models/wired_xlstm.py` (usage)

For historical context, see:
- `docs_archive/components/mad/MAD_WIRING_INTEGRATION.md` (old approach)
- `docs_archive/architecture/XLSTM_MAD_NCPS_DESIGN.md` (design evolution)
