# Engineering Spec: The MAD Wiring System

Hello everyone! If you're diving into our codebase, you'll quickly encounter a system we call **MAD**, which stands for
**M**odular **A**rchitecture **D**escription. This document is your guide to understanding what it is, why we use it,
and how you can work with it.

## The "Why": Moving Beyond Sequential Chains

Most deep learning models are defined as a simple, sequential stack of layers. You might write something like this in
PyTorch or MLX:

```python
# The "classic" way
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(...)
        self.lstm1 = nn.LSTM(...)
        self.lstm2 = nn.LSTM(...)
        self.output = nn.Linear(...)

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.output(x)
        return x
```

This is great for simple models, but it has limitations:

1. **It's Rigid:** The architecture is hardcoded. What if you want to experiment with a residual connection from `lstm1`
   to the `output`? Or add a normalization layer in between? You have to change the `forward` method every time.
2. **It's Opaque:** The `forward` method *is* the definition of the architecture. To understand the model's structure,
   you have to read Python code, which can become complex.
3. **It Hides Parallelism:** It's not immediately obvious which operations could run in parallel.

MAD is our solution to these challenges. It allows us to define a model's architecture **declaratively**, as a graph of
connected blocks, separate from the execution logic.

## The "What": Core Concepts of MAD

MAD is built on a few key ideas. You'll see these reflected in the `xlstm_metal/wiring/mlx/wiring.py` file.

1. **Blocks (`BlockSpec`)**: A "block" is a fundamental unit of computation. It could be an embedding layer, an xLSTM
   block, a normalization layer, or even a non-neural component like a tokenizer. We define each block with a
   `BlockSpec`, which is like a blueprint. It specifies the block's `type`, its `parameters` (like `embedding_dim`), and
   its `backend` (e.g., `MLX`).

2. **Wiring (`MADWiring`)**: This is the master blueprint for the entire model. It holds two things:
    * A collection of all the `BlockSpec`s in the model.
    * An **adjacency matrix** that defines how these blocks are connected. It's a directed graph where an edge from
      block A to block B means A's output is an input to B.

3. **The Runner (`WiredMADModel`)**: This is the engine that brings the declarative wiring to life. It's an
   `mlx.nn.Module` that:
    * Takes a `MADWiring` object in its constructor.
    * Instantiates the actual MLX modules (like `nn.Linear`, `xLSTMBlock`) for each `BlockSpec`.
    * Determines the correct execution order using a **topological sort** of the graph. This sort groups blocks into "
      stages" that can be executed in parallel.
    * The `__call__` (forward pass) method iterates through these stages, gathers inputs for each block, executes it,
      and caches its output for downstream blocks to use.

This separation of concerns is the superpower of MAD:

* `MADWiring` **describes** the graph (the "what").
* `WiredMADModel` **executes** the graph (the "how").

## A Deeper Dive: The `WiredMADModel` Forward Pass

Let's walk through how `WiredMADModel` in `wiring.py` executes a forward pass. This is where the magic happens.

```python
# A simplified view of the forward pass in WiredMADModel
def __call__(self, x, hidden_states=None):
    activations = {} # Cache for block outputs
    new_hidden_states = {}

    # self.stages is a list of lists, e.g., [['embedding'], ['lstm_1', 'lstm_2'], ['norm'], ['lm_head']]
    for stage_blocks in self.stages:
        
        # In a real parallel implementation, blocks in a stage could run concurrently.
        for block_name in stage_blocks:
            # 1. Gather Inputs
            incoming_connections = self.wiring.get_connections(block_name, 'incoming')
            if not incoming_connections:
                # This is the entry point of the graph
                block_input = x
            else:
                # Aggregate inputs from parent blocks
                inputs = []
                for source_name in incoming_connections:
                    # The wiring can even specify a "polarity" (+1 or -1) for a connection,
                    # allowing for subtractive interactions (like in gating mechanisms).
                    polarity = self.wiring.get_polarity(source_name, block_name)
                    scaled_input = polarity * activations[source_name]
                    inputs.append(scaled_input)
                
                # Combine inputs, typically by summation
                block_input = mx.stack(inputs).sum(axis=0)

            # 2. Execute Block
            block = self._blocks[block_name]
            block_hidden = hidden_states.get(block_name)
            output = block(block_input, block_hidden) # Or just block(block_input)

            # 3. Handle State and Cache Output
            if isinstance(output, tuple): # Stateful blocks like xLSTM return (output, new_state)
                output, new_hidden = output
                new_hidden_states[block_name] = new_hidden
            
            activations[block_name] = output

    # 4. Return Final Output
    return activations[self.output_block_name], new_hidden_states
```

### Key Takeaways from the Forward Pass

* **Dynamic Input Aggregation**: The model isn't limited to a single input. A block can sum or otherwise combine outputs
  from any number of preceding blocks. This is perfect for residual connections, merges, and more complex topologies.
* **State Management**: The wiring system automatically handles passing hidden states to and from stateful blocks (like
  our `xLSTMBlock`), making it easy to build recurrent architectures.
* **Parallelism by Design**: The concept of "stages" explicitly defines the potential for parallelism. All blocks within
  a single stage are independent of each other and can be executed concurrently, which is a huge advantage for
  performance on multi-core hardware.

---

## Implementation Deep Dive: From Config to Execution

The conceptual overview is powerful, but the real magic is in the implementation. Here's how the `xlstm_metal` library
turns a configuration file into an executable model.

### Step 1: Creating the Wiring (`create_xlstm_wiring`)

Everything starts in `xlstm_metal/wiring/mlx/xlstm_7b.py` with the `create_xlstm_wiring` function. This function takes a
`config` dictionary (loaded from `config.json`) and builds the `MADWiring` graph.

It creates `BlockSpec` objects for each component of the standard xLSTM architecture:

1. **`embedding`**: A `BlockType.EMBEDDING` block (`mlx.nn.Embedding`).
2. **`xlstm_{i}`**: A series of `BlockType.MLSTM` blocks (our custom `xLSTMBlock`).
3. **`out_norm`**: A `BlockType.NORM` block (`RMSNorm`).
4. **`lm_head`**: A `BlockType.LINEAR` block (`mlx.nn.Linear`) for the final output projection.

It then connects them in a simple, sequential chain:
`embedding` → `xlstm_0` → ... → `xlstm_{n-1}` → `out_norm` → `lm_head`

### Step 2: Instantiating Blocks in `WiredMADModel`

When the `WiredMADModel` is initialized with this wiring, its `_instantiate_block` method maps each `BlockSpec` to a
concrete `mlx.nn.Module`:

- `BlockType.MLSTM` becomes an `xlstm_metal.blocks.mlstm_mlx.xlstm_block.xLSTMBlock`.
- `BlockType.NORM` becomes an `xlstm_metal.blocks.mlstm_mlx.components.RMSNorm`.
- Other types map to standard `mlx.nn` layers.

This is the bridge from the abstract `BlockSpec` to the actual code that will be executed.

## Anatomy of the Core Building Blocks

The power of MAD is that these blocks can be arbitrarily complex. Let's break down the most important ones.

### The `xLSTMBlock`

The `xLSTMBlock` (in `xlstm_block.py`) is the heart of the model. It is **not** a single layer, but a composite block
with two parallel branches, each with its own pre-normalization and residual connection.

```
Input
  |
  +---> RMSNorm ---> mLSTMLayer ---> (+) ---> Output
  |      (xlstm_norm)      (xlstm)      |
  |                                     |
  +---> RMSNorm ---> GatedFFN ------> (+) ---> Output
         (ffn_norm)        (ffn)
```

The input `x` is fed to both branches simultaneously. The output of each branch is added back to the original input
`x` (residual connection). This structure is what you see implemented in the `xLSTMBlock.__call__` method.

### The `mLSTMLayer`

The `mLSTMLayer` (in `block.py`) is the implementation of the novel recurrent core. It's a complex module with several
key features:

- **Projections**: It starts by projecting the input into **Q, K, V** vectors and three separate **gate** vectors (
  input, forget, output).
- **Gate Soft-Capping**: To improve stability, the pre-activations for the input and forget gates are capped to a
  maximum value (e.g., 15.0).
- **Dynamic Kernel Selection**: The layer intelligently chooses a backend kernel based on the sequence length (`S`) for
  maximum performance:
    - **`S == 1`**: For single-token generation (like in autoregressive sampling), it uses a highly optimized, purely
      recurrent step (`mlstm_recurrent_step`).
    - **`1 < S < chunk_size`**: For short sequences, it runs a Python loop over the recurrent step (`mlstm_sequential`).
    - **`S >= chunk_size`**: For long sequences (like processing a prompt), it uses a `mlstm_chunkwise` parallel kernel
      that processes the sequence in chunks, combining recurrence within chunks and parallelism between them.
- **Multi-Head Normalization**: It uses a custom `MultiHeadLayerNorm` that normalizes the output of each head
  *independently* before it's passed to the output gate. This is a key difference from standard layer normalization.
- **Gated Output**: The final output is produced by an element-wise multiplication of the normalized hidden state and
  the output gate (`h_out = sigmoid(o_preact) * h_norm`), which is then passed through a final output projection.

### The `GatedFFN`

The `GatedFFN` (in `ffn.py`) is a SwiGLU-style feed-forward network. Instead of a simple linear projection, it uses a
gating mechanism to control the flow of information:

1. The input `x` is projected into two separate matrices: a `gate` and a value `z`.
2. The `gate` is passed through an activation function (SiLU, also known as Swish).
3. The final output is the element-wise product of the activated `gate` and `z`, which is then projected back down to
   the embedding dimension.
    - `gated = SiLU(gate) * z`
    - `y = proj_down(gated)`

This allows the network to dynamically modulate which features pass through the FFN.

## Tying It All Together: From Inference to Wiring

When you use the `xLSTMRunner.from_pretrained(...)` in `xlstm_metal.inference.generate`, you are seeing the entire MAD
system in action:

1. The runner downloads the model and loads its `config.json`.
2. It calls `create_xlstm_wiring(config)` to build the declarative `MADWiring` graph.
3. It instantiates a `WiredMADModel`, passing it the newly created wiring.
4. When you call `runner.generate()`, the `WiredMADModel` executes the graph as described above, token by token.

The MAD system provides a robust, flexible, and high-performance foundation for our models. By understanding these core
components—the declarative wiring, the execution engine, and the anatomy of the blocks—you can effectively navigate,
modify, and extend our xLSTM architecture.
