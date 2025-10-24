# MAD Wiring Integration: NCPS-Style Composition for xLSTM

**Using sparse wiring patterns for explicit parallel structure and weight sharing**

Date: 2025-01-21
Source: NCPS wiring.py, LTC implementation patterns
Goal: Replace backends.py with declarative wiring for MAD blocks

## The Key Insight

**NCPS wiring provides:**
1. **Explicit connectivity** via adjacency matrices
2. **Sparse patterns** for efficient parallel computation
3. **Weight sharing** through shared connections
4. **Declarative composition** (no imperative backend selection)

**This is exactly what MAD needs** to move beyond sequential layer stacking!

## Current MAD vs Wiring-Based MAD

### Current Approach (Sequential)
```python
# mad/registry.py
layers = ['mlstm', 'swiglu', 'mlstm', 'swiglu']

# Sequential execution
for layer_name in layers:
    x = layer_registry[layer_name](x)
```

**Problems:**
- No parallelism
- No weight sharing
- Backend selection is global
- Can't express "bricks" patterns

### Wiring-Based Approach (Graph)
```python
# Define connectivity graph
wiring = MADWiring(
    blocks={
        'mlstm_1': mLSTMBlock(heads=8),
        'mlstm_2': mLSTMBlock(heads=8),
        'swiglu': SwiGLUBlock(),
        'attention': AttentionBlock(),
    }
)

# Define connections (adjacency matrix)
wiring.add_connection('mlstm_1', 'swiglu', polarity=1)
wiring.add_connection('mlstm_2', 'swiglu', polarity=1)  # Parallel!
wiring.add_connection('swiglu', 'attention', polarity=1)

# Execution respects graph structure (parallel where possible)
output = wiring(input)
```

## NCPS Wiring Patterns for MAD

### 1. **FullyConnected Wiring** → Dense Block Connections

```python
class DenseMADWiring(Wiring):
    """Fully connected MAD blocks (every block connects to every other)"""

    def __init__(self, blocks: List[str]):
        super().__init__(units=len(blocks))
        self.block_names = blocks

        # Fully connected between blocks
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                if i != j:  # No self-connections
                    self.add_synapse(i, j, polarity=1)
```

**Use case:** Dense architectures like early ResNets, DenseNet

### 2. **Random Wiring** → Sparse Architecture Search

```python
class SparseMADWiring(Wiring):
    """Randomly sparse connections (for architecture search)"""

    def __init__(self, blocks: List[str], sparsity=0.7, seed=42):
        super().__init__(units=len(blocks))
        self.sparsity = sparsity

        # Randomly connect blocks
        rng = PyRandom(seed)
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                if i != j and rng.random() > sparsity:
                    self.add_synapse(i, j, polarity=1)
```

**Use case:** MAD architecture search, finding optimal sparse patterns

### 3. **NCP Wiring** → Layered xLSTM Architecture

```python
class xLSTMWiring(Wiring):
    """
    Three-layer wiring for xLSTM:
    - Inter layer: mLSTM blocks (memory)
    - Command layer: sLSTM blocks (control)
    - Motor layer: FFN blocks (output)
    """

    def __init__(
        self,
        num_mlstm_blocks: int,
        num_slstm_blocks: int,
        num_ffn_blocks: int,
        mlstm_to_slstm_fanout: int = 2,
        slstm_to_ffn_fanout: int = 1,
    ):
        total = num_mlstm_blocks + num_slstm_blocks + num_ffn_blocks
        super().__init__(units=total)

        # Define layers
        self.mlstm_neurons = list(range(num_mlstm_blocks))
        self.slstm_neurons = list(range(
            num_mlstm_blocks,
            num_mlstm_blocks + num_slstm_blocks
        ))
        self.ffn_neurons = list(range(
            num_mlstm_blocks + num_slstm_blocks,
            total
        ))

        # Connect mLSTM → sLSTM (with fanout)
        for mlstm_idx in self.mlstm_neurons:
            targets = random.sample(self.slstm_neurons, mlstm_to_slstm_fanout)
            for slstm_idx in targets:
                self.add_synapse(mlstm_idx, slstm_idx, polarity=1)

        # Connect sLSTM → FFN (with fanout)
        for slstm_idx in self.slstm_neurons:
            targets = random.sample(self.ffn_neurons, slstm_to_ffn_fanout)
            for ffn_idx in targets:
                self.add_synapse(slstm_idx, ffn_idx, polarity=1)
```

**Use case:** Hierarchical memory (mLSTM) + control (sLSTM) + output (FFN)

### 4. **ParallelHeads Wiring** → Multi-Head Parallelism

```python
class MultiHeadWiring(Wiring):
    """
    Parallel attention heads with cross-head connections

    Structure:
    [Head 0] ─┐
    [Head 1] ─┼─> [Combiner]
    [Head 2] ─┘
    """

    def __init__(self, num_heads: int, cross_head_connections: bool = False):
        # num_heads + 1 combiner block
        super().__init__(units=num_heads + 1)

        self.head_neurons = list(range(num_heads))
        self.combiner_neuron = num_heads

        # All heads feed into combiner
        for head_idx in self.head_neurons:
            self.add_synapse(head_idx, self.combiner_neuron, polarity=1)

        # Optional: Cross-head connections (for head interaction)
        if cross_head_connections:
            for i in self.head_neurons:
                for j in self.head_neurons:
                    if i != j:
                        self.add_synapse(i, j, polarity=1)
```

**Use case:** Parallel attention heads, MoE routing

## MAD Block Wiring Implementation

### Core Wiring Class

```python
# mad/wiring.py
from typing import Dict, List, Callable
import mlx.core as mx

class MADWiring(Wiring):
    """
    Wiring for MAD blocks with:
    - Explicit connectivity graph
    - Weight sharing via shared edges
    - Parallel execution where possible
    """

    def __init__(self, block_specs: Dict[str, Dict]):
        """
        Args:
            block_specs: Dict mapping block names to specs
                {
                    'mlstm_1': {
                        'type': 'mlstm',
                        'backend': 'mlx',
                        'params': {'num_heads': 8}
                    },
                    'swiglu': {
                        'type': 'swiglu',
                        'backend': 'mlx'
                    }
                }
        """
        super().__init__(units=len(block_specs))

        self.block_specs = block_specs
        self.block_names = list(block_specs.keys())
        self.name_to_idx = {name: idx for idx, name in enumerate(self.block_names)}

        # Instantiate blocks
        self.blocks = {}
        for name, spec in block_specs.items():
            block_type = spec['type']
            backend = spec.get('backend', 'mlx')
            params = spec.get('params', {})

            # Get block class from registry
            block_class = self._get_block_class(block_type, backend)
            self.blocks[name] = block_class(**params)

    def add_block_connection(
        self,
        src_block: str,
        dest_block: str,
        polarity: int = 1,
        weight_sharing: bool = False
    ):
        """Add connection between two blocks"""
        src_idx = self.name_to_idx[src_block]
        dest_idx = self.name_to_idx[dest_block]

        self.add_synapse(src_idx, dest_idx, polarity)

        # Track weight sharing
        if weight_sharing:
            self._mark_weight_sharing(src_idx, dest_idx)

    def _get_execution_order(self) -> List[List[int]]:
        """
        Compute execution order respecting dependencies

        Returns list of "stages" where blocks in same stage
        can execute in parallel
        """
        # Topological sort with level assignment
        in_degree = mx.sum(mx.abs(self.adjacency_matrix), axis=0)

        stages = []
        remaining = set(range(self.units))

        while remaining:
            # Find blocks with no unprocessed dependencies
            stage = [i for i in remaining if in_degree[i] == 0]

            if not stage:
                raise ValueError("Cyclic dependency detected in wiring")

            stages.append(stage)

            # Remove these blocks
            for idx in stage:
                remaining.remove(idx)
                # Decrease in-degree of successors
                for successor in range(self.units):
                    if self.adjacency_matrix[idx, successor] != 0:
                        in_degree[successor] -= 1

        return stages

    def __call__(self, x: mx.array, states: Dict = None) -> mx.array:
        """
        Execute wiring graph

        Parallelizes blocks within same stage
        """
        if states is None:
            states = {name: None for name in self.block_names}

        execution_order = self._get_execution_order()

        # Track outputs
        block_outputs = {}

        for stage in execution_order:
            # All blocks in this stage can run in PARALLEL
            stage_outputs = []

            for block_idx in stage:
                block_name = self.block_names[block_idx]
                block = self.blocks[block_name]

                # Gather inputs from predecessors
                predecessors = [
                    i for i in range(self.units)
                    if self.adjacency_matrix[i, block_idx] != 0
                ]

                if not predecessors:
                    # Input block
                    block_input = x
                else:
                    # Combine predecessor outputs
                    inputs = [block_outputs[self.block_names[i]] for i in predecessors]
                    block_input = self._combine_inputs(inputs, predecessors, block_idx)

                # Execute block (can be parallelized across stage)
                output, state = block(block_input, states[block_name])

                block_outputs[block_name] = output
                states[block_name] = state

        # Return output from final block(s)
        final_blocks = [
            name for idx, name in enumerate(self.block_names)
            if mx.sum(mx.abs(self.adjacency_matrix[idx, :])) == 0  # No outgoing edges
        ]

        if len(final_blocks) == 1:
            return block_outputs[final_blocks[0]], states
        else:
            # Multiple outputs - concatenate
            return mx.concatenate([block_outputs[name] for name in final_blocks], axis=-1), states

    def _combine_inputs(
        self,
        inputs: List[mx.array],
        pred_indices: List[int],
        dest_idx: int
    ) -> mx.array:
        """
        Combine multiple inputs based on polarity

        Positive polarity: add
        Negative polarity: subtract
        """
        combined = inputs[0]

        for i, pred_idx in enumerate(pred_indices[1:], 1):
            polarity = self.adjacency_matrix[pred_idx, dest_idx]
            if polarity > 0:
                combined = combined + inputs[i]
            else:
                combined = combined - inputs[i]

        return combined
```

### Usage Example

```python
# mad/models/wired_language_model.py
class WiredLanguageModel(mx.nn.Module):
    """MAD language model using explicit wiring"""

    def __init__(self, config):
        super().__init__()

        # Define blocks
        block_specs = {
            # Parallel mLSTM heads
            'mlstm_head_0': {
                'type': 'mlstm',
                'backend': 'mlx',
                'params': {'num_heads': 2, 'head_dim': 64}
            },
            'mlstm_head_1': {
                'type': 'mlstm',
                'backend': 'mlx',
                'params': {'num_heads': 2, 'head_dim': 64}
            },
            'mlstm_head_2': {
                'type': 'mlstm',
                'backend': 'mlx',
                'params': {'num_heads': 2, 'head_dim': 64}
            },
            'mlstm_head_3': {
                'type': 'mlstm',
                'backend': 'mlx',
                'params': {'num_heads': 2, 'head_dim': 64}
            },

            # Combiner
            'head_combiner': {
                'type': 'linear',
                'backend': 'mlx',
                'params': {'in_features': 512, 'out_features': 512}
            },

            # FFN
            'swiglu': {
                'type': 'swiglu',
                'backend': 'mlx',
                'params': {'d_model': 512}
            }
        }

        # Create wiring
        self.wiring = MADWiring(block_specs)

        # Parallel structure: All heads run simultaneously
        for i in range(4):
            self.wiring.add_block_connection(
                f'mlstm_head_{i}',
                'head_combiner',
                polarity=1
            )

        # Sequential: Combiner → FFN
        self.wiring.add_block_connection('head_combiner', 'swiglu', polarity=1)

        # Embedding/output
        self.embedding = mx.nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = mx.nn.Linear(config.d_model, config.vocab_size)

    def __call__(self, input_ids, states=None):
        x = self.embedding(input_ids)

        # Execute wired graph (parallelizes where possible)
        x, new_states = self.wiring(x, states)

        logits = self.lm_head(x)
        return logits, new_states
```

## Porting Triton to Metal

The Triton kernels can be ported to Metal using MLX's `mx.fast.metal_kernel`:

```python
# mad/blocks/mlstm_mlx/kernels/metal_tfla.py
import mlx.core as mx

def metal_chunkwise_forward(
    q: mx.array,  # [batch, seq, num_heads, head_dim]
    k: mx.array,
    v: mx.array,
    i_gate: mx.array,  # [batch, seq, num_heads]
    f_gate: mx.array,
    chunk_size: int = 64
) -> tuple[mx.array, tuple]:
    """
    Metal implementation of TFLA (Tiled Flash Linear Attention)

    Ports chunkwise--triton_xl_chunk to Metal
    """

    # Metal shader source
    metal_source = """
    kernel void chunkwise_forward(
        device const float* q [[buffer(0)]],
        device const float* k [[buffer(1)]],
        device const float* v [[buffer(2)]],
        device const float* i_gate [[buffer(3)]],
        device const float* f_gate [[buffer(4)]],
        device float* output [[buffer(5)]],
        device float* C_out [[buffer(6)]],
        device float* n_out [[buffer(7)]],
        device float* m_out [[buffer(8)]],
        constant int& batch_size [[buffer(9)]],
        constant int& seq_len [[buffer(10)]],
        constant int& num_heads [[buffer(11)]],
        constant int& head_dim [[buffer(12)]],
        constant int& chunk_size [[buffer(13)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        // Thread indices
        int batch_idx = gid.x;
        int head_idx = gid.y;
        int chunk_idx = gid.z;

        if (batch_idx >= batch_size || head_idx >= num_heads) return;

        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = min(chunk_start + chunk_size, seq_len);

        // Initialize state for this chunk
        float C[HEAD_DIM][HEAD_DIM];  // Covariance matrix
        float n[HEAD_DIM];             // Normalizer
        float m = -INFINITY;           // Running max

        // Load previous state if not first chunk
        if (chunk_idx > 0) {
            // Load from previous chunk's final state
            // ... (state loading logic)
        } else {
            // Initialize to zeros
            for (int i = 0; i < HEAD_DIM; i++) {
                n[i] = 0.0f;
                for (int j = 0; j < HEAD_DIM; j++) {
                    C[i][j] = 0.0f;
                }
            }
        }

        // Process chunk
        for (int t = chunk_start; t < chunk_end; t++) {
            // Get inputs for this timestep
            float q_t[HEAD_DIM], k_t[HEAD_DIM], v_t[HEAD_DIM];
            float i_t = i_gate[batch_idx * seq_len * num_heads + t * num_heads + head_idx];
            float f_t = f_gate[batch_idx * seq_len * num_heads + t * num_heads + head_idx];

            // Load q, k, v
            for (int d = 0; d < HEAD_DIM; d++) {
                int offset = batch_idx * seq_len * num_heads * HEAD_DIM +
                             t * num_heads * HEAD_DIM +
                             head_idx * HEAD_DIM + d;
                q_t[d] = q[offset];
                k_t[d] = k[offset];
                v_t[d] = v[offset];
            }

            // Update m_t (running max)
            m = max(m + f_t, i_t);

            // Exponential gates (stabilized)
            float i_exp = exp(i_t - m);
            float f_exp = exp(f_t + m_prev - m);

            // Update covariance: C_t = f * C_{t-1} + i * (v ⊗ k)
            for (int i = 0; i < HEAD_DIM; i++) {
                for (int j = 0; j < HEAD_DIM; j++) {
                    C[i][j] = f_exp * C[i][j] + i_exp * v_t[i] * k_t[j];
                }
            }

            // Update normalizer: n_t = f * n_{t-1} + i * k
            for (int d = 0; d < HEAD_DIM; d++) {
                n[d] = f_exp * n[d] + i_exp * k_t[d];
            }

            // Compute output: h_t = (C_t @ q_t) / (n_t · q_t)
            float numerator[HEAD_DIM];
            for (int i = 0; i < HEAD_DIM; i++) {
                numerator[i] = 0.0f;
                for (int j = 0; j < HEAD_DIM; j++) {
                    numerator[i] += C[i][j] * q_t[j];
                }
            }

            float denominator = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                denominator += n[d] * q_t[d];
            }
            denominator = max(denominator, 1.0f);  // Clamp to prevent division by zero

            // Write output
            for (int d = 0; d < HEAD_DIM; d++) {
                int out_offset = batch_idx * seq_len * num_heads * HEAD_DIM +
                                 t * num_heads * HEAD_DIM +
                                 head_idx * HEAD_DIM + d;
                output[out_offset] = numerator[d] / denominator;
            }

            m_prev = m;
        }

        // Store final state for next chunk
        // ... (state storage logic)
    }
    """

    # Compile Metal kernel
    kernel = mx.fast.metal_kernel(
        name="chunkwise_forward",
        input_names=["q", "k", "v", "i_gate", "f_gate"],
        output_names=["output", "C_out", "n_out", "m_out"],
        source=metal_source
    )

    # Allocate outputs
    batch, seq_len, num_heads, head_dim = q.shape
    output = mx.zeros_like(q)
    C_out = mx.zeros((batch, num_heads, head_dim, head_dim))
    n_out = mx.zeros((batch, num_heads, head_dim))
    m_out = mx.zeros((batch, num_heads))

    # Launch kernel
    grid = (batch, num_heads, (seq_len + chunk_size - 1) // chunk_size)
    outputs = kernel(
        inputs=[q, k, v, i_gate, f_gate],
        grid=grid,
        threadgroup=(1, 1, 1),
        output_shapes=[(batch, seq_len, num_heads, head_dim), C_out.shape, n_out.shape, m_out.shape]
    )

    return outputs[0], (outputs[1], outputs[2], outputs[3])
```

## Summary: MAD Without backends.py

**Replace:**
```python
# backends.py (imperative backend selection)
if backend == 'mlx':
    use_mlx_block()
elif backend == 'pytorch':
    use_pytorch_block()
```

**With:**
```python
# wiring.py (declarative graph composition)
wiring = MADWiring(block_specs)
wiring.add_block_connection('mlstm_1', 'combiner')
wiring.add_block_connection('mlstm_2', 'combiner')  # Parallel!
output = wiring(input)  # Automatically parallelizes
```

**Benefits:**
1. **Explicit parallelism** - graph structure shows what runs in parallel
2. **Weight sharing** - shared edges = shared weights
3. **No backend.py** - blocks know their own implementation
4. **NCPS patterns** - reuse proven wiring strategies
5. **Metal kernels** - Port Triton TFLA to Metal for MLX

This is the "real parallelism" AND the compositional flexibility MAD needs!