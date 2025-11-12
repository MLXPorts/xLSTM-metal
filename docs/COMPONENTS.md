# xLSTM-Metal Component Documentation

**Author:** Sydney Renee (sydney@solace.ofharmony.ai)  
**Extracted from:** Production docstrings, kernel comments, and implementation notes  
**Last Updated:** January 2025

This document dissects the component hierarchy of xLSTM-metal, a production-quality implementation of xLSTM for Apple Silicon. Every piece here went through real Metal kernel development, numerical stability battles, and MLX integration challenges. The docstrings aren't theoretical—they reflect actual working code paths and design decisions made under fire.

If you're looking for high-level architecture, see `ARCHITECTURE.md`. This is the component-by-component reference.

## Navigation

This document follows the natural call hierarchy: top-level model → blocks → cells → kernels.

**High-Level:**
- [Model Architecture](#model) - WiredxLSTM and weight loading
- [NCPS Wiring](#ncps-wiring) - Automatic structure discovery
- [Utilities](#utilities) - Config loading, safetensors handling

**Core Blocks:**
- [mLSTM Block](#mlstm-block) - Matrix-memory LSTM implementation
- [sLSTM Block](#slstm) - Scalar-memory LSTM implementation
- [FFN](#ffn) - SwiGLU feed-forward networks

**mLSTM Internals:**
- [mLSTM Cells](#mlstm-cells) - Projection, kernel, output cells
- [mLSTM Forward Kernels](#mlstm-forward-kernels) - Metal chunkwise computation
- [mLSTM Backward Kernels](#mlstm-backward-kernels) - Gradient kernels (training)

**sLSTM Internals:**
- [sLSTM Layers](#slstm) - Projection, kernel, output cells
- [sLSTM Conv](#slstm-conv) - Causal convolution preprocessing

**Shared Components:**
- [Normalization](#normalization) - RMSNorm, MultiHeadNorm
- [Other](#other) - Soft-cap, utility functions

## FFN

### SwiGLU: The Modern Feed-Forward Standard

Every xLSTM block ends with a gated FFN using the SwiGLU pattern (from PaLM, LLaMA). This replaced the old ReLU FFN years ago and we're never going back.

**Traditional FFN:**
```
FFN(x) = W₂ · ReLU(W₁ · x)
```

**SwiGLU (what we use):**
```
FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)
```

The gating mechanism (element-wise multiply after activation) provides multiplicative interactions that improve expressiveness without adding depth. SiLU (Swish) has smoother gradients than ReLU, which matters for deep networks.

**Parameter count:** Same as a 2-layer FFN with larger hidden size. Typical expansion: 2.667× for xLSTM-7B.

### `blocks/mlstm/ffn/gated_ffn_cell.py`

**Stateless transformation cell.** This is the actual computation—three linear layers (gate, up, down) with SiLU activation on the gate path. No recurrence, no state, just pure feedforward.

**Usage in xLSTM blocks:**
```python
residual = x
x = layer_norm(x)
x = mLSTM(x)  # or sLSTM
x = x + residual

residual = x
x = layer_norm(x)
x = GatedFFN(x)
x = x + residual
```

Two residual connections per block (post-norm pattern). This is critical for gradient flow in deep stacks.

### `blocks/mlstm/ffn/gated_ffn.py`

**NCPS module wrapper** around the cell. Handles sequence iteration for frameworks that expect an RNN-like interface. In practice, for xLSTM we use the cell directly since we process entire sequences at once—this wrapper is for NCPS compatibility when wiring expects module-level APIs.


[MODULE] gated_ffn
------------------------------------------------------------
Gated Feed-Forward Network Module – MLX Implementation (NCPS Sequence Wrapper)

Overview
--------
GatedFFN is a sequence-processing wrapper around GatedFFNCell, following
the NCPS pattern where:
  - **Cell**: Single-step computation with parameters (GatedFFNCell)
  - **Module**: Batch/sequence handling wrapper (this class)

This separation enables flexible integration with NCPS wiring while maintaining
compatibility with standard sequence-to-sequence interfaces.

NCPS Module Pattern
-------------------
Similar to how RNN modules wrap RNN cells:
  - Cell: processes one timestep at a time
  - Module: iterates over sequence dimension, calling cell repeatedly

For stateless FFN, this pattern seems redundant but maintains API consistency
with stateful cells (LSTM, GRU, CfC) in NCPS frameworks.

Sequence Processing Modes
--------------------------
1. **return_sequences=True** (default):
   - Returns all timestep outputs [B, S, D]
   - Used for encoder-style processing

2. **return_sequences=False**:
   - Returns only last timestep [B, D]
   - Used for sequence classification

Batch Dimension Ordering
-------------------------
- **batch_first=True** (default): Input shape [B, S, D]
  - Standard PyTorch/MLX convention
  - B = batch size, S = sequence length, D = features

- **batch_first=False**: Input shape [S, B, D]
  - Legacy RNN convention (rarely used in modern code)

When to Use This vs GatedFFNCell Directly?
-------------------------------------------
Use **GatedFFNCell** when:
  - You have already-batched single-step inputs [B, D]
  - You're building custom sequence processing logic
  - You want minimal overhead

Use **GatedFFN** when:
  - You need standard sequence-to-sequence interface
  - You want compatibility with NCPS wiring infrastructure
  - You need return_sequences or batch_first options

In Practice
-----------
For xLSTM blocks, FFN is typically applied to entire sequences at once
(not iteratively), so direct GatedFFNCell usage is more efficient:

  # Efficient (used in xLSTM blocks)
  ffn_cell = GatedFFNCell(input_size, hidden_size)
  output, _ = ffn_cell(x)  # x: [B, S, D]

  # Equivalent but slower (iterates over sequence)
  ffn_module = GatedFFN(input_size, hidden_size)
  output, _ = ffn_module(x)  # x: [B, S, D]

This wrapper is primarily for NCPS-style applications where the wiring
framework expects a module-level interface.

Optional Projection
-------------------
If `proj_size` is specified, an additional linear layer projects the output
to a different dimension. This is useful for encoder-decoder architectures
or when embedding dimension differs from model dimension.

Stateless Property
------------------
Unlike LSTM/GRU, FFN has no hidden state. The `hx` parameter and return
value are kept for API compatibility but are always None.

Parity
------
Logic mirrors torch-native GatedFFN for cross-backend testing.

[CLASS] GatedFFN
------------------------------------------------------------
Sequence-processing wrapper for GatedFFNCell (NCPS module pattern).

Handles batch and sequence dimensions, iterates over timesteps calling
the underlying cell. Primarily for NCPS framework compatibility.

Parameters
----------
input_size : int
    Input dimension (embedding_dim).
hidden_size : int
    Intermediate dimension for FFN (~2.667 * input_size typical).
proj_size : int | None, optional
    Optional output projection dimension (default: None = input_size).
return_sequences : bool, default True
    Whether to return all timesteps (True) or only last (False).
batch_first : bool, default True
    Whether input shape is [B, S, D] (True) or [S, B, D] (False).
activation : {"silu", "gelu", "relu"}, default "silu"
    Activation function for gating.
use_bias : bool, default False
    Whether linear layers include bias.
dropout : float | None, optional
    Dropout probability for regularization.

Returns (forward)
-----------------
output : mx.array
    - If return_sequences=True: [B, S, output_size] (or [S, B, output_size])
    - If return_sequences=False: [B, output_size] (or [output_size])
state : None
    Always None (FFN is stateless).

Examples
--------
>>> ffn = GatedFFN(input_size=512, hidden_size=1365)
>>> x = mx.random.normal((4, 32, 512))  # [B, S, D]
>>> y, state = ffn(x)
>>> y.shape
(4, 32, 512)
>>> state is None
True

[FUNCTION] _apply_fc
------------------------------------------------------------
Apply optional projection layer.

[FUNCTION] __call__
------------------------------------------------------------
Process sequence through gated FFN.

Parameters
----------
inputs : mx.array
    Input sequences [B, S, D] if batch_first else [S, B, D].
hx : mx.array | None, optional
    Hidden state (unused, kept for API compatibility).

Returns
-------
output : mx.array
    Processed sequences (shape depends on return_sequences).
state : None
    Always None (stateless).

Notes
-----
Iterates over sequence dimension, applying cell at each timestep.
For efficiency, consider using GatedFFNCell directly on full
sequences when return_sequences=True and no special processing needed.



### `blocks/mlstm/ffn/gated_ffn_cell.py`


[MODULE] gated_ffn_cell
------------------------------------------------------------
Gated Feed-Forward Network Cell – MLX Implementation (SwiGLU Pattern)

Overview
--------
GatedFFNCell implements the SwiGLU (Swish-Gated Linear Unit) feed-forward
network, a modern variant of FFN that uses gating to control information flow.
This is the standard FFN architecture in LLaMA, PaLM, and xLSTM models.

SwiGLU Architecture
-------------------
Traditional FFN:
  FFN(x) = W₂ · ReLU(W₁ · x)

SwiGLU (Gated FFN):
  FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)

where ⊙ denotes element-wise multiplication and SiLU(x) = x · σ(x).

The gating mechanism allows the network to learn which features to amplify
or suppress, improving expressiveness without additional depth.

Why Gated FFN?
--------------
1. **Better Expressiveness**: Gating provides multiplicative interactions
2. **Smoother Gradients**: SiLU has smoother derivatives than ReLU
3. **Empirical Performance**: SwiGLU outperforms ReLU FFN in LLMs
4. **Parameter Efficiency**: Same param count as 2-layer FFN with larger hidden

NCPS Pattern
------------
This cell follows the NCPS (Neural Circuit Policies) pattern:
  - Cell = single-step computation with all trainable parameters
  - Module wraps cell for sequence/batch processing
  - Stateless: returns (output, None) since FFN has no hidden state

Typical Usage in xLSTM
-----------------------
In xLSTM blocks, GatedFFN is applied after mLSTM/sLSTM:
  x = layer_norm(x)
  x = mLSTM(x)
  x = x + residual

  residual = x
  x = layer_norm(x)
  x = GatedFFN(x)
  x = x + residual

Activation Functions
--------------------
Supported activations:
  - **silu** (default): SiLU(x) = x · σ(x), smooth and non-monotonic
  - **gelu**: GELU(x) ≈ x · Φ(x), used in BERT/GPT
  - **relu**: ReLU(x) = max(0, x), traditional choice

SiLU is the standard for SwiGLU and provides best empirical results.

Sparsity Mask
-------------
Optional `sparsity_mask` enables structured sparsity patterns for NCPS
wiring. When provided, the mask is applied element-wise to the gated hidden
states, zeroing out specific connections.

Dropout
-------
Optional dropout is applied to the final output for regularization during
training. Typically disabled for inference.

Parameters vs FLOPs
-------------------
For embedding_dim D and hidden_size H:
  - Parameters: 3 * D * H (gate, up, down projections)
  - FLOPs: ~6 * D * H * S (forward pass for sequence length S)

Typical hidden_size is 2.667 * embedding_dim (xLSTM-7B uses this ratio).

Parity
------
Logic mirrors torch-native GatedFFNCell for cross-backend testing.

[CLASS] GatedFFNCell
------------------------------------------------------------
SwiGLU-style gated FFN cell (stateless, single-step).

Implements gated feed-forward transformation with configurable activation.
Follows NCPS cell pattern: encapsulates all parameters, single-step forward.

Parameters
----------
input_size : int
    Input/output dimension (embedding_dim).
hidden_size : int
    Intermediate dimension (typically ~2.667 * input_size).
activation : {"silu", "gelu", "relu"}, default "silu"
    Activation function for gating.
use_bias : bool, default False
    Whether linear layers include bias (typically False for xLSTM).
dropout : float | None, optional
    Dropout probability for output (training regularization).
sparsity_mask : mx.array | None, optional
    Optional mask for structured sparsity (NCPS wiring).

Returns (forward)
-----------------
output : mx.array [B, S, input_size]
    Transformed features.
state : None
    Always None (FFN is stateless).

Examples
--------
>>> cell = GatedFFNCell(input_size=4096, hidden_size=10880)
>>> x = mx.random.normal((2, 64, 4096))
>>> y, state = cell(x)
>>> y.shape
(2, 64, 4096)
>>> state is None
True

[FUNCTION] __call__
------------------------------------------------------------
Apply SwiGLU gated FFN transformation.

Parameters
----------
x : mx.array [B, S, input_size]
    Input features.
state : mx.array | None, optional
    Unused (kept for NCPS API compatibility with stateful cells).

Returns
-------
output : mx.array [B, S, input_size]
    Gated FFN output.
state : None
    Always None (stateless).

Notes
-----
Computation flow:
  1. gate = act(W_gate @ x)
  2. up = W_up @ x
  3. hidden = gate ⊙ up
  4. output = W_down @ hidden

[FUNCTION] get_config
------------------------------------------------------------
Return configuration for serialization.



---

## Model

### `models/wired_xlstm.py`

The top-level language model class. This is what you import when you want to run inference or fine-tune. It handles automatic architecture discovery from safetensors, builds the block stack dynamically, and manages weight loading without hardcoded configurations.

**Key insight:** Traditional LLM loaders require separate classes for 1B, 7B, 13B variants. WiredxLSTM introspects the checkpoint structure and builds the correct model on the fly. One class, any xLSTM size.

**Architecture Stack:**
```
Token IDs [B, S]
  ↓ embedding
Embeddings [B, S, D]
  ↓ blocks[0..N-1] (mLSTM/sLSTM/attention mix)
Hidden [B, S, D]
  ↓ out_norm (RMSNorm)
Normalized [B, S, D]
  ↓ lm_head (Linear)
Logits [B, S, vocab_size]
```

Each block maintains recurrent state (C, n, m) for autoregressive generation. State reuse enables efficient token-by-token generation without reprocessing context.


[MODULE] wired_xlstm
------------------------------------------------------------
**Production model-agnostic xLSTM implementation**

This isn't your typical hardcoded transformer class. WiredxLSTM discovers architecture from checkpoint files and assembles the correct block types dynamically. Works with any xLSTM variant (1B, 7B, 13B+) without code changes.

**Model-Agnostic Design Philosophy:**
- Parse safetensors index to discover structure
- Build block cells (mLSTM/sLSTM/attention) on demand
- Single codebase handles all model sizes
- Automatic weight mapping via introspection

**Complete Stack:**

  Input Token IDs [B, S]
    ↓ embedding
  Embeddings [B, S, D]
    ↓ blocks[0..N-1]
  Hidden [B, S, D]
    ↓ out_norm (RMSNorm)
  Normalized [B, S, D]
    ↓ lm_head (Linear)
  Logits [B, S, vocab_size]

Each block is typically:
  residual = x
  x = norm_mlstm(x)
  x, state = mlstm_cell(x, state)
  x = x + residual

  residual = x
  x = norm_ffn(x)
  x = ffn(x)
  x = x + residual

Automatic Structure Discovery
------------------------------
WiredxLSTM uses AutoWiring to:
  1. Parse model.safetensors.index.json
  2. Detect block types (mlstm_layer, slstm_layer, attn_layer)
  3. Count blocks and identify special layers (embedding, out_norm, lm_head)
  4. Build sequential connectivity
  5. Create appropriate cell instances for each block

Weight Loading
--------------
Weights are loaded from safetensors shards using the mapping:
  - backbone.embeddings.weight → embedding layer
  - backbone.blocks.{i}.* → block[i] parameters
  - backbone.out_norm.weight → output normalization
  - lm_head.weight → language model head

Each block provides get_weight_keys() to map internal parameters to
safetensors keys, enabling automatic weight loading without hardcoding
parameter names.

Weight Tying
------------
When tie_word_embeddings=True (common for LLMs to reduce parameters):
  lm_head.weight = embedding.weight
This shares the embedding matrix for both input and output, reducing
total parameters by vocab_size * embedding_dim.

Mixed Precision
---------------
Supports independent dtypes:
  - compute_dtype: Forward pass activations (bfloat16 for speed)
  - state_dtype: Recurrent state storage (float32 for stability)
  - param_dtype: Trainable parameters (inherits from compute_dtype)

Typical inference config: compute=bfloat16, state=float32

Stateful Generation
-------------------
The model maintains recurrent state across tokens during generation:

  # Initial prompt
  logits, states = model(prompt_ids, state=None, return_last_states=True)

  # Autoregressive generation
  for _ in range(max_new_tokens):
      next_id = sample(logits[:, -1, :])
      logits, states = model(next_id.reshape(1,1), state=states, return_last_states=True)

State reuse enables efficient generation without reprocessing context.

Configuration Sources
---------------------
Model configuration comes from multiple sources:
  1. **config.json**: Base hyperparameters (embedding_dim, num_heads, etc.)
  2. **Safetensors index**: Derived structure (num_blocks, block types)
  3. **Runtime kwargs**: Dtype overrides, mode settings

The wiring object combines these into a complete runtime config.

Usage Patterns
--------------
Automatic loading (recommended):
  >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
  >>> logits = model(input_ids)

Manual wiring (advanced):
  >>> wiring = create_auto_wiring("xlstm_7b_model")
  >>> model = WiredxLSTM(wiring=wiring, load_weights=True)

Custom dtype:
  >>> model = WiredxLSTM.from_pretrained(
  ...     "xlstm_7b_model",
  ...     compute_dtype=mx.bfloat16,
  ...     state_dtype=mx.float32
  ... )

Special Tokens
--------------
Model respects tokenizer special tokens from config:
  - pad_token_id: Padding token (attention masking)
  - bos_token_id: Beginning-of-sequence token
  - eos_token_id: End-of-sequence token
  - force_bos_token_insert: Auto-prepend BOS if missing

Embedding Dropout
-----------------
Optional dropout on embeddings (typically disabled for inference):
  add_embedding_dropout=True, embedding_dropout=0.1

Training vs Inference Mode
--------------------------
Model defaults to eval() when config['mode']='inference' to:
  - Disable dropout
  - Use deterministic behavior
  - Enable efficient generation

Call .train() to enable dropout for fine-tuning.

Extensibility
-------------
Adding new block types:
  1. Implement block cell class (e.g., AttentionBlock)
  2. Update detect_block_type() in auto_wiring.py
  3. Add case in create_block_cell() factory method

WiredxLSTM automatically handles new types without modification.

Parity
------
Logic mirrors torch-native WiredxLSTM for cross-backend compatibility.

[CLASS] WiredxLSTM
------------------------------------------------------------
Model-agnostic xLSTM language model with automatic structure discovery.

Assembles complete LM from safetensors using NCPS wiring. Supports
mixed block types (mLSTM, sLSTM, attention) with automatic weight loading.

Parameters
----------
wiring : AutoWiring
    Wiring object defining model structure (from create_auto_wiring).
load_weights : bool, default False
    Whether to load pretrained weights from safetensors.
model_dir : str | Path | None, optional
    Model directory (for weight loading, defaults to wiring.model_dir).
compute_dtype : mx.Dtype, default mx.float32
    Dtype for forward pass activations.
state_dtype : mx.Dtype, default mx.float32
    Dtype for recurrent state storage.
norm_reduce_force_float32 : bool, default True
    Force float32 in norm reductions for stability.

Attributes
----------
wiring : AutoWiring
    Model structure specification.
config : dict
    Complete configuration (from config.json + derived values).
blocks : list
    Stack of xLSTM blocks (mLSTM/sLSTM/attention cells).
embedding : nn.Embedding | None
    Token embedding layer.
out_norm : RMSNormCell | None
    Pre-LM-head normalization (uses custom RMSNormCell for canonical behavior).
lm_head : nn.Linear | None
    Language modeling head (vocab projection).

Methods
-------
__call__(input_ids, state, return_last_states)
    Forward pass through model.
from_pretrained(model_dir, **kwargs)
    Class method to load from checkpoint directory.
load_pretrained_weights()
    Load weights from safetensors files.

Examples
--------
>>> # Automatic loading
>>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
>>> logits = model(input_ids)

>>> # Stateful generation
>>> logits, states = model(prompt, return_last_states=True)
>>> for _ in range(100):
...     next_token = sample(logits[:, -1, :])
...     logits, states = model(next_token.reshape(1,1), state=states, return_last_states=True)

[FUNCTION] _build_model
------------------------------------------------------------
Build model architecture from wiring specification.

[FUNCTION] __call__
------------------------------------------------------------
Forward pass through complete language model.

Parameters
----------
input_ids : mx.array [B, S]
    Input token IDs.
state : list of tuple | None, optional
    List of recurrent states for each block (for stateful generation).
return_last_states : bool, default False
    Whether to return final recurrent states.

Returns
-------
logits : mx.array [B, S, vocab_size]
    Output logits for next-token prediction.
states : list of tuple (optional)
    Final recurrent states for each block (if return_last_states=True).

Notes
-----
- Embedding dropout applied if configured
- Each block processes sequentially with residual connections
- Output norm applied before LM head
- States enable efficient autoregressive generation

[FUNCTION] load_pretrained_weights
------------------------------------------------------------
Load pretrained weights from safetensors files.

[FUNCTION] _load_weights_from_dict
------------------------------------------------------------
Map safetensors weights to model parameters.

[FUNCTION] from_pretrained
------------------------------------------------------------
Load complete model from HuggingFace checkpoint directory.

Discovers architecture from safetensors, builds model, loads weights.

Parameters
----------
model_dir : str | Path
    Path to model directory with config.json and safetensors files.
load_weights : bool, default True
    Whether to load pretrained weights.
**kwargs
    Additional arguments for WiredxLSTM constructor
    (compute_dtype, state_dtype, etc.).

Returns
-------
model : WiredxLSTM
    Initialized model ready for inference or fine-tuning.

Examples
--------
>>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
>>> model = WiredxLSTM.from_pretrained(
...     "xlstm_7b_model",
...     compute_dtype=mx.bfloat16,
...     state_dtype=mx.float32
... )

[FUNCTION] get_config
------------------------------------------------------------
Return model configuration.

[FUNCTION] _apply_weight_tying
------------------------------------------------------------
Tie LM head weights to embeddings when the config requests it.



## NCPS Wiring

### `wiring/auto_wiring.py`

**The discovery engine.** This is what makes model-agnostic loading possible. Instead of hardcoding "xLSTM-7B has 32 mLSTM blocks with these dimensions," AutoWiring reads the safetensors index file and figures out the architecture automatically.

**Discovery Process:**
1. Parse `model.safetensors.index.json` for weight keys
2. Extract block indices and component types (mlstm_layer, slstm_layer, etc.)
3. Build sequential connectivity graph
4. Provide factory methods for cell instantiation

**Why this matters:** You can train a custom xLSTM variant with mixed block types, different dimensions, whatever—and this loader will just work. No code changes needed.

**Block Type Detection:**
- `mlstm_layer` → Matrix-memory LSTM (attention-like)
- `slstm_layer` → Scalar-memory LSTM (traditional RNN-like)
- `attn_layer` → Standard attention (future support)

The wiring object becomes a blueprint that WiredxLSTM uses to build the actual model.


[MODULE] auto_wiring
------------------------------------------------------------
Automatic Wiring Generation – MLX Implementation (Model Structure Discovery)

Overview
--------
AutoWiring automatically discovers model architecture from safetensors files
and generates appropriate NCPS wiring. This enables **model-agnostic loading**:
instead of hardcoding model structure, the system introspects checkpoint files
and builds the correct cell types and connectivity.

Problem Statement
-----------------
Traditional model loading requires:
  1. Knowing the exact architecture (layer types, counts, dimensions)
  2. Writing custom loading code for each model variant
  3. Maintaining separate codebases for 1B, 7B, 13B, etc. models

AutoWiring solves this by:
  - Parsing safetensors index JSON to discover blocks and components
  - Detecting block types (mLSTM, sLSTM, attention) from weight keys
  - Creating appropriate cell instances with correct configurations
  - Building sequential wiring automatically

How It Works
------------
1. **Structure Analysis**
   - Read `model.safetensors.index.json`
   - Extract all weight keys (e.g., `backbone.blocks.0.mlstm_layer.q.weight`)
   - Group by block index to find num_blocks
   - Detect special components (embedding, out_norm, lm_head)

2. **Block Type Detection**
   - If block has `mlstm_layer` → mLSTMBlock
   - If block has `slstm_layer` → sLSTMBlock
   - If block has `attn_layer` → AttentionBlock (future)

3. **Wiring Construction**
   - Create sequential connectivity: block_0 → block_1 → ... → block_N
   - Set units = num_blocks + special_layers
   - Provide `create_block_cell()` factory method

4. **Cell Creation**
   - Each block calls `.from_config(block_idx, config)` on appropriate class
   - Config loaded from `config.json` provides hyperparameters
   - Cells manage their own weight loading via `get_weight_keys()`

Usage Pattern
-------------
```python
# Automatic discovery
wiring = create_auto_wiring("xlstm_7b_model")
print(f"Detected {wiring.structure['num_blocks']} blocks")

# Create appropriate cell for each block
cells = [wiring.create_block_cell(i) for i in range(wiring.structure['num_blocks'])]

# Or use with WiredxLSTM wrapper
model = WiredxLSTM(wiring=wiring, load_weights=True)
```

Structure Dictionary
--------------------
`wiring.structure` contains:
  - `num_blocks`: int - Number of transformer/LSTM blocks
  - `block_components`: {idx: [component_names]} - Per-block component list
  - `has_embedding`: bool - Whether model has token embedding layer
  - `has_out_norm`: bool - Whether model has pre-LM-head normalization
  - `has_lm_head`: bool - Whether model has language modeling head

Block Type Detection
--------------------
Component patterns:
  - `mlstm_layer` → matrix-memory LSTM (xLSTM-7B default)
  - `slstm_layer` → scalar-memory LSTM (alternative variant)
  - `attn_layer` or `attention` → standard attention (future support)

Sequential Wiring
-----------------
For xLSTM models, wiring is always sequential (each block feeds next).
Future extensions could support:
  - Mixture-of-experts (sparse routing)
  - Skip connections (ResNet-style)
  - Parallel branches (ensemble-like)

Benefits
--------
1. **Zero-config loading**: Works with any xLSTM checkpoint
2. **Version agnostic**: Adapts to checkpoint structure changes
3. **Modular**: Easy to add new block types
4. **Introspectable**: Query block types before instantiation
5. **Portable**: Same pattern works across MLX/PyTorch backends

Parity
------
Logic mirrors torch-native AutoWiring for cross-backend compatibility.

[FUNCTION] _parse_block_index
------------------------------------------------------------
Parse a string containing a valid JSON integer (e.g., '0', '15') as an int.

This function uses `json.loads()` instead of `int()` to ensure that only
valid JSON integer representations are accepted (e.g., no floats, hex, or
other non-JSON formats). This provides type safety and strict validation
of block indices as they appear in JSON-based model index files.

[FUNCTION] analyze_safetensors_structure
------------------------------------------------------------
Discover model architecture from safetensors index file.

Parses `model.safetensors.index.json` to extract block structure,
component types, and special layers (embedding, norm, LM head).

Parameters
----------
model_dir : str
    Path to model directory containing safetensors files.

Returns
-------
structure : dict
    Model structure dictionary with keys:
    - `num_blocks`: Number of transformer/LSTM blocks
    - `block_components`: {block_idx: [component_names]}
    - `has_embedding`: Whether model has token embeddings
    - `has_out_norm`: Whether model has pre-head normalization
    - `has_lm_head`: Whether model has language modeling head

Raises
------
FileNotFoundError
    If safetensors index not found in model_dir.

Example
-------
>>> structure = analyze_safetensors_structure("xlstm_7b_model")
>>> print(structure)
{
    'num_blocks': 32,
    'block_components': {0: ['ffn', 'mlstm_layer', 'norm_ffn', 'norm_mlstm'], ...},
    'has_embedding': True,
    'has_out_norm': True,
    'has_lm_head': True
}

[FUNCTION] detect_block_type
------------------------------------------------------------
Infer block type from component names.

Examines component list to determine whether block is mLSTM, sLSTM,
attention, or unknown type.

Parameters
----------
components : list of str
    Component names from safetensors keys (e.g., ['mlstm_layer', 'ffn', 'norm_mlstm']).

Returns
-------
block_type : {"mlstm", "slstm", "attention", "unknown"}
    Detected block type.

Examples
--------
>>> detect_block_type(['mlstm_layer', 'ffn', 'norm_mlstm'])
'mlstm'
>>> detect_block_type(['slstm_layer', 'ffn'])
'slstm'

[CLASS] AutoWiring
------------------------------------------------------------
Automatic wiring with model structure discovery and cell factory.

Extends base Wiring with automatic architecture detection from safetensors
and factory methods for creating appropriate block cells.

Parameters
----------
model_dir : str
    Path to model directory with safetensors and config.json.
config : dict | None, optional
    Model configuration (loaded from config.json if not provided).

Attributes
----------
structure : dict
    Discovered model structure (from `analyze_safetensors_structure`).
block_types : dict
    Mapping {block_idx: block_type_str}.
config : dict
    Model hyperparameters from config.json.
model_dir : Path
    Path to model directory.

Methods
-------
get_block_info(block_idx)
    Query block type and components.
create_block_cell(block_idx, **kwargs)
    Factory method to instantiate appropriate cell for block.

[FUNCTION] create_auto_wiring
------------------------------------------------------------
Main entry point for automatic model loading.

Discovers model structure from safetensors and creates wiring with
cell factory methods. This is the recommended way to load xLSTM models.

Parameters
----------
model_dir : str
    Path to model directory with safetensors and config.json.
config : dict | None, optional
    Model configuration (loaded automatically if not provided).

Returns
-------
wiring : AutoWiring
    AutoWiring instance with discovered structure.

Example
-------
>>> # Automatic loading (most common)
>>> wiring = create_auto_wiring("xlstm_7b_model")
>>> model = WiredxLSTM(wiring=wiring, load_weights=True)

>>> # Custom config override
>>> custom_config = load_config("xlstm_7b_model")
>>> custom_config['chunk_size'] = 128
>>> wiring = create_auto_wiring("xlstm_7b_model", custom_config)

[FUNCTION] __init__
------------------------------------------------------------
Create automatic wiring from model directory.

Args:
    model_dir: Path to model directory with safetensors
    config: Optional config dict (loaded from config.json)

[FUNCTION] _build_connections
------------------------------------------------------------
Build sequential connections between blocks.

[FUNCTION] get_block_info
------------------------------------------------------------
Query information about a specific block.

Parameters
----------
block_idx : int
    Block index [0, num_blocks).

Returns
-------
block_info : dict
    Dictionary with keys:
    - `index`: Block index
    - `type`: Block type string
    - `components`: List of component names

Example
-------
>>> info = wiring.get_block_info(0)
>>> print(info['type'])
'mlstm'

[FUNCTION] create_block_cell
------------------------------------------------------------
Factory method to create appropriate cell for block.

Detects block type and instantiates corresponding class (mLSTMBlock,
sLSTMBlock, etc.) with configuration from config.json.

Parameters
----------
block_idx : int
    Block index to create cell for.
**kwargs
    Additional arguments passed to cell constructor.

Returns
-------
cell : nn.Module
    Instantiated cell (mLSTMBlock, sLSTMBlock, etc.).

Raises
------
NotImplementedError
    If block type not yet supported.
ValueError
    If block type unknown/invalid.

Example
-------
>>> cell = wiring.create_block_cell(0, compute_dtype=mx.bfloat16)
>>> isinstance(cell, mLSTMBlock)
True



### `wiring/wirings.py`


[MODULE] wirings
------------------------------------------------------------
NCPS Wiring – MLX Implementation (Sparse Neural Circuit Blueprints)

Overview
--------
Wiring defines the **connectivity blueprint** for sparse neural circuits in
the NCPS (Neural Circuit Policies) framework. Instead of densely connecting
all neurons, wiring specifies which neurons connect to which (analogous to
biological synaptic connectivity patterns).

NCPS Philosophy
---------------
Traditional neural networks use dense weight matrices where every input
connects to every output. NCPS instead defines:
  - **Neurons**: Computational units (e.g., LSTM cells, attention heads)
  - **Synapses**: Directed connections between neurons with polarity
  - **Wiring**: The adjacency matrix encoding which synapses exist

This modularity enables:
  1. Compositional architectures (mix different cell types)
  2. Sparse connectivity (reduce parameters, improve interpretability)
  3. Biologically-inspired circuit motifs (feed-forward, recurrent, lateral inhibition)

Neuron Types
------------
- **Sensory**: Input neurons receiving external features
- **Inter**: Internal/hidden neurons (computation, memory)
- **Motor**: Output neurons producing predictions/actions

Synapse Polarity
---------------
Each synapse has polarity ∈ {-1, +1}:
  - **Excitatory (+1)**: Positive influence (strengthen activation)
  - **Inhibitory (-1)**: Negative influence (suppress activation)

In xLSTM context, polarity is typically +1 (standard feed-forward flow).
Inhibitory synapses can model gating or competition between pathways.

Adjacency Matrices
------------------
Two connectivity matrices:
  1. **adjacency_matrix**: [units, units] inter-neuron connections
  2. **sensory_adjacency_matrix**: [input_dim, units] sensory → inter connections

Entry values:
  - 0: No synapse
  - +1: Excitatory synapse
  - -1: Inhibitory synapse

Sequential Wiring Pattern (xLSTM)
---------------------------------
For transformer/LSTM-style models, wiring is typically **sequential**:
  block_0 → block_1 → block_2 → ... → block_N → output

Each block is a "neuron" in NCPS terminology, and sequential connectivity
ensures information flows through the entire stack.

Usage in xLSTM
--------------
While xLSTM models use sequential stacking (not sparse wiring), the Wiring
abstraction provides:
  - Uniform interface for model assembly
  - Introspection (query block types, connectivity)
  - Extensibility (future sparse/mixture variants)

Building Wiring
---------------
1. Create wiring: `wiring = Wiring(units=32)`
2. Add synapses: `wiring.add_synapse(src=0, dest=1, polarity=1)`
3. Set input: `wiring.build(input_dim=256)`
4. Add sensory: `wiring.add_sensory_synapse(src=0, dest=5, polarity=1)`

Serialization
-------------
Wiring can be saved/loaded via `get_config()` / `from_config()` for
reproducibility and transfer across frameworks.

Visualization
-------------
The `draw_graph()` method renders the wiring as a NetworkX graph for
inspection and debugging.

Parity
------
Logic mirrors torch-native Wiring for cross-backend compatibility.

[CLASS] Wiring
------------------------------------------------------------
Connectivity blueprint for sparse neural circuits (NCPS framework).

Encodes which neurons connect to which via adjacency matrices. Supports
both inter-neuron (recurrent) and sensory (input) connections. Provides
serialization, visualization, and introspection methods.

Parameters
----------
units : int
    Number of neurons in the circuit (excluding sensory inputs).

Attributes
----------
adjacency_matrix : mx.array [units, units]
    Inter-neuron connectivity matrix (0 = no synapse, ±1 = synapse polarity).
sensory_adjacency_matrix : mx.array [input_dim, units] | None
    Sensory → inter-neuron connectivity (set after build()).
input_dim : int | None
    Number of sensory input features (set via build()).
output_dim : int | None
    Number of motor output neurons (set via set_output_dim()).

Methods
-------
build(input_dim)
    Initialize sensory connectivity matrix.
add_synapse(src, dest, polarity)
    Add inter-neuron synapse.
add_sensory_synapse(src, dest, polarity)
    Add sensory → inter-neuron synapse.
get_config() / from_config(config)
    Serialize/deserialize wiring for persistence.
draw_graph(layout, colors, labels)
    Visualize wiring as NetworkX graph (requires matplotlib).

[FUNCTION] num_layers
------------------------------------------------------------
:return:

[FUNCTION] get_neurons_of_layer
------------------------------------------------------------
:param layer_id:
:return:

[FUNCTION] is_built
------------------------------------------------------------
:return:

[FUNCTION] build
------------------------------------------------------------
Initialize wiring with input dimension (creates sensory adjacency).

Parameters
----------
input_dim : int
    Number of sensory input features.

Raises
------
ValueError
    If input_dim conflicts with previously set value.

[FUNCTION] erev_initializer
------------------------------------------------------------
:param _:
:param __:
:return:

[FUNCTION] sensory_erev_initializer
------------------------------------------------------------
:param _:
:param __:
:return:

[FUNCTION] set_input_dim
------------------------------------------------------------
:param input_dim:

[FUNCTION] set_output_dim
------------------------------------------------------------
:param output_dim:

[FUNCTION] get_type_of_neuron
------------------------------------------------------------
Classify neuron as 'motor' (output) or 'inter' (hidden).

Parameters
----------
neuron_id : int
    Neuron index.

Returns
-------
neuron_type : {"motor", "inter"}
    Neuron classification based on output_dim.

[FUNCTION] add_synapse
------------------------------------------------------------
Add inter-neuron synapse (recurrent connection).

Parameters
----------
src : int
    Source neuron index [0, units).
dest : int
    Destination neuron index [0, units).
polarity : {-1, +1}, default 1
    Synapse polarity (excitatory +1, inhibitory -1).

Raises
------
ValueError
    If indices out of bounds or invalid polarity.

[FUNCTION] add_sensory_synapse
------------------------------------------------------------
Add sensory → inter-neuron synapse (input connection).

Parameters
----------
src : int
    Sensory input index [0, input_dim).
dest : int
    Destination neuron index [0, units).
polarity : {-1, +1}, default 1
    Synapse polarity.

Raises
------
ValueError
    If wiring not built or indices out of bounds.

[FUNCTION] get_config
------------------------------------------------------------
:return:

[FUNCTION] from_config
------------------------------------------------------------
:param config:
:return:

[FUNCTION] get_graph
------------------------------------------------------------
:param include_sensory_neurons:
:return:

[FUNCTION] draw_graph
------------------------------------------------------------
Render wiring as NetworkX graph (requires matplotlib).

Parameters
----------
layout : str, default "shell"
    NetworkX layout algorithm: {"kamada", "circular", "spring", "spectral", ...}
neuron_colors : dict | None
    Mapping {neuron_type: color} for node coloring.
synapse_colors : dict | None
    Mapping {polarity: color} for edge coloring.
draw_labels : bool, default False
    Whether to annotate nodes with neuron IDs.

Returns
-------
legend_patches : list
    Matplotlib patch objects for legend creation.

[FUNCTION] print_diagram
------------------------------------------------------------
Print a textual wiring diagram with simple ASCII/Unicode glyphs.

[FUNCTION] synapse_count
------------------------------------------------------------
:return:

[FUNCTION] sensory_synapse_count
------------------------------------------------------------
:return:


## Normalization

### RMSNorm: The Numerical Stability Workhorse

RMS normalization is everywhere in this codebase: pre-mLSTM, pre-FFN, inside mLSTM output cells, you name it. It's simpler than LayerNorm (no mean centering) but performs just as well for transformers.

**Why RMSNorm over LayerNorm?**
- Single reduction pass (sum of squares only)
- No bias correction needed
- ~15% faster, same quality in practice
- Metal kernel friendly (fewer synchronization points)

**Force Float32 Reductions:**
The `force_float32_reductions=True` flag is **critical** for mixed-precision inference. When running bfloat16 activations, accumulating sum-of-squares in bfloat16 causes precision drift over long reductions (4096+ dims). Forcing float32 accumulation fixes this without impacting throughput.

### `blocks/rms_norm/rmsnorm.py`

**Metal-accelerated implementation** using custom kernels for Apple Silicon. The kernel parallelizes reduction across threadgroups with shared memory for partial sums.

**Kernel Strategy:**
- Thread count: min(256, feature_dim)
- Each thread accumulates partial sum over assigned columns
- Threadgroup barrier for synchronization
- Thread 0 computes final RMS
- All threads apply scaling + weight multiplication

### `blocks/mlstm/multihead_norm/multihead_norm.py`

**Per-head normalization** for mLSTM output processing. Critical insight: different attention heads operate at different scales. Global normalization swamps individual head statistics. Per-head norm computes statistics independently for each head's features before flattening.

**Tensor flow:**
```
Input:  [B, S, NH, DH]  (multi-head format)
  ↓ Normalize each head [:, :, h, :] independently
Normalized: [B, S, NH, DH]
  ↓ Flatten to [B, S, NH * DH]
Output: [B, S, NH * DH]  (ready for projection)
```

**Weight shape detail:** Weight is stored as flat `[NH * DH]`, NOT `[NH, DH]`. This matches HuggingFace transformers xlstm and enables per-feature rescaling after per-head normalization.


[MODULE] multihead_norm
------------------------------------------------------------
Multi-Head Normalization Layers – MLX Implementation

Overview
--------
Multi-head normalization applies layer normalization or RMS normalization
**independently per head** rather than across the entire flattened dimension.
This is critical for multi-head architectures (like mLSTM) where different
heads may operate at different activation scales.

Why Per-Head Normalization?
----------------------------
In standard LayerNorm/RMSNorm over a flattened [NH * DH] dimension, the
normalization statistics (mean, variance, RMS) are computed globally. This
can be problematic when:
  - One head dominates in magnitude (its stats swamp others)
  - Different heads encode different types of information at different scales
  - Gradient flow becomes imbalanced across heads

Per-head normalization computes stats independently for each head's DH
features, ensuring each head is normalized to a consistent scale before
being combined.

Tensor Flow
-----------
Input:  [B, S, NH, DH]  (multi-head format)
  ↓ Normalize each [:, :, h, :] independently
Normalized: [B, S, NH, DH]
  ↓ Flatten to [B, S, NH * DH]
Output: [B, S, NH * DH]  (ready for projection)

Weight Shape
------------
**CRITICAL**: Weight is stored as a **flat vector [NH * DH]**, not [NH, DH].
This matches the HuggingFace transformers xLSTM implementation and allows
a single learnable parameter per feature dimension.

After per-head normalization and flattening, the weight is applied:
  output = normalized_flat * weight

This design enables the model to learn per-feature rescaling while maintaining
per-head normalization statistics.

MultiHeadLayerNorm vs MultiHeadRMSNorm
--------------------------------------
- **LayerNorm**: Computes mean and variance per head, normalizes via
  (x - mean) / sqrt(var + eps). More compute (two passes) but zero-centers.

- **RMSNorm**: Computes only RMS = sqrt(mean(x²) + eps), normalizes via
  x / RMS. Single pass, no mean centering. Often equivalent performance.

Force Float32 Reductions
------------------------
When `force_float32_reductions=True`, mean/variance/RMS computations are
performed in float32 even if inputs are bfloat16. This prevents accumulation
errors in long reductions (large DH) and is **strongly recommended** for
stable mixed-precision training.

Usage in mLSTM
--------------
The mLSTM output cell uses MultiHeadRMSNorm to normalize hidden states h
before applying the output gate and projection:

  h: [B, NH, S, DH_v] → transpose → [B, S, NH, DH_v]
  h_norm = MultiHeadRMSNorm(h)  → [B, S, NH * DH_v]
  output = LinearProjection(h_norm * output_gate)

This ensures each head's contribution is properly scaled before the final
projection back to embedding_dim.

Parity
------
Logic mirrors torch-native MultiHeadLayerNorm/MultiHeadRMSNorm for testing.

[CLASS] MultiHeadLayerNorm
------------------------------------------------------------
Per-head LayerNorm with flattening (mean centering + variance scaling).

Applies standard LayerNorm independently to each head's features, then
flattens and applies a shared weight vector. Commonly used when zero-
centering is beneficial for downstream layers.

Parameters
----------
num_heads : int
    Number of attention heads (NH).
head_dim : int
    Dimension per head (DH).
eps : float, default 1e-6
    Numerical stability epsilon for variance.
use_weight : bool, default True
    Whether to apply learnable weight scaling.
use_bias : bool, default False
    Whether to apply learnable bias (after normalization).

Returns (forward)
-----------------
output : mx.array [B, S, NH * DH]
    Normalized and flattened activations.

Notes
-----
Weight/bias are flat [NH * DH], applied **after** per-head normalization
and flattening (matches HuggingFace xLSTM design).

[CLASS] MultiHeadRMSNorm
------------------------------------------------------------
Per-head RMSNorm with flattening (RMS scaling only, no mean centering).

Applies RMS normalization independently to each head's features, then
flattens and applies a shared weight vector. Preferred for efficiency
when mean centering is not required.

Per-Head RMS Computation
------------------------
For each head h:
  RMS_h = sqrt(mean(x[:,:,h,:]²) + eps)
  x_norm[:,:,h,:] = x[:,:,h,:] / RMS_h

Parameters
----------
num_heads : int
    Number of attention heads (NH).
head_dim : int
    Dimension per head (DH).
eps : float, default 1e-6
    Numerical stability epsilon for RMS.
use_weight : bool, default True
    Whether to apply learnable weight scaling.
force_float32_reductions : bool, default True
    Force float32 accumulation in RMS computation (recommended).

Returns (forward)
-----------------
output : mx.array [B, S, NH * DH]
    RMS-normalized and flattened activations.

Notes
-----
Weight is flat [NH * DH], applied **after** per-head RMS normalization
and flattening. This is the standard pattern in xLSTM output cells.

[FUNCTION] __call__
------------------------------------------------------------
Normalize per head, flatten, and scale.

Parameters
----------
x : mx.array [B, S, NH, DH]
    Multi-head input tensor.

Returns
-------
output : mx.array [B, S, NH * DH]
    Normalized and flattened output.

[FUNCTION] __call__
------------------------------------------------------------
RMS normalize per head, flatten, and scale.

Parameters
----------
x : mx.array [B, S, NH, DH]
    Multi-head input tensor.

Returns
-------
output : mx.array [B, S, NH * DH]
    RMS-normalized and flattened output.



### `blocks/rms_norm/rmsnorm.py`


[MODULE] rmsnorm
------------------------------------------------------------
Metal-Accelerated RMS Normalization – MLX Implementation

Overview
--------
Root Mean Square (RMS) normalization is a simplified variant of layer
normalization that omits mean centering and only rescales by the RMS:

  RMSNorm(x) = (x / RMS(x)) * weight
  where RMS(x) = sqrt(mean(x²) + eps)

This is computationally cheaper than full LayerNorm (no mean subtraction,
no variance bias correction) and empirically performs similarly for
transformer-style models.

Why RMSNorm?
------------
- **Efficiency**: One pass (sum of squares), no mean centering.
- **Stability**: The additive epsilon prevents division by zero.
- **Performance**: In large models (e.g., LLaMA, xLSTM), RMSNorm matches
  LayerNorm quality with ~15% less compute.

Metal Acceleration
------------------
This implementation uses a custom Metal kernel for Apple Silicon that:
  1. Parallelizes sum-of-squares reduction across threads in a threadgroup
  2. Uses shared memory (threadgroup memory) for partial sums
  3. Supports mixed precision (float32 accumulation with bfloat16 I/O)
  4. Fuses RMS computation and weight scaling into a single kernel launch

Force Float32 Reductions
------------------------
When `force_float32_reductions=True`, the kernel accumulates squared values
in float32 even if inputs are bfloat16. This prevents precision loss in
long reductions (e.g., dims=4096) where bfloat16's limited mantissa can
cause drift.

Multi-Head Variants
-------------------
The module also provides `MultiHeadRMSNormCell` which applies RMSNorm
independently per head (useful for mLSTM output processing). The per-head
statistics ensure each head's activations are normalized separately before
being flattened and projected.

Usage
-----
Standard RMSNorm for backbone layers:
  norm = RMSNormCell(dims=4096, eps=1e-6, force_float32_reductions=True)
  out = norm(x)  # x: [B, S, 4096]

Multi-head variant (for mLSTM output):
  norm = MultiHeadRMSNormCell(num_heads=8, head_dim=512, eps=1e-6)
  out = norm(x)  # x: [B, S, 8, 512] -> [B, S, 4096]

Parity
------
Logic mirrors torch-native RMSNorm implementations for cross-backend testing.

[CLASS] RMSNormMetalKernel
------------------------------------------------------------
Reusable Metal kernel for RMSNorm with dtype-specific compilation.

Compiles and caches Metal shaders on-demand based on input dtype and
precision settings. The kernel implements a parallel reduction for
computing mean(x²) followed by element-wise rescaling.

Kernel Strategy
---------------
- Thread count per row: min(256, cols)
- Each thread accumulates partial sum over its assigned columns
- Threadgroup barrier synchronizes partial sums
- Thread 0 reduces threadgroup partials to compute RMS
- All threads apply RMS scaling and weight multiplication

Parameters
----------
None (stateless, kernel cache is instance attribute)

Methods
-------
build(dtype, force_float32) -> metal_kernel
    Compile and return cached kernel for given dtype/precision.
apply(inputs_2d, weight, eps, force_float32) -> output_2d
    Execute kernel on 2D-reshaped input.

[CLASS] RMSNormCell
------------------------------------------------------------
NCPS-style RMSNorm cell using Metal-accelerated kernel.

Wraps the Metal kernel with NCPS-compatible interface (stateless,
composable). Handles arbitrary input shapes by flattening to 2D,
applying the kernel, and reshaping back.

Parameters
----------
dims : int
    Feature dimension (last axis of input).
eps : float, default 1e-6
    Numerical stability epsilon.
use_weight : bool, default True
    Whether to apply learnable weight scaling.
force_float32_reductions : bool, default True
    Force float32 accumulation in reduction (recommended for bfloat16).
kernel : RMSNormMetalKernel | None, optional
    Custom kernel instance (default creates new).
debug_compare : bool | None, optional
    Enable torch reference comparison (for validation).
param_dtype : mx.Dtype, default mx.float32
    Dtype for weight parameter.

Returns (forward)
-----------------
output : mx.array, same shape as input
    RMS-normalized and weight-scaled activations.

[CLASS] MultiHeadRMSNormCell
------------------------------------------------------------
Multi-head RMSNorm applying per-head normalization then flattening.

Normalizes each head independently over head_dim, then reshapes to
[B, S, NH * DH] and applies a shared flat weight. Used in mLSTM
output cells where per-head statistics improve stability.

Per-Head Normalization
----------------------
For input [B, S, NH, DH]:
  1. Compute RMS(x[:,:,h,:]) for each head h
  2. Normalize: x_norm[:,:,h,:] = x[:,:,h,:] / RMS_h
  3. Flatten: [B, S, NH, DH] -> [B, S, NH*DH]
  4. Scale: x_norm * weight (weight is flat [NH*DH])

Why Per-Head?
-------------
Different heads may learn different activation scales. Independent
normalization prevents one head from dominating and improves gradient
flow across heads.

Parameters
----------
num_heads : int
    Number of attention heads (NH).
head_dim : int
    Dimension per head (DH).
eps : float, default 1e-6
    Numerical stability epsilon.
force_float32_reductions : bool, default True
    Force float32 accumulation in RMS computation.
kernel : RMSNormMetalKernel | None, optional
    Shared kernel instance (default creates new).
param_dtype : mx.Dtype, default mx.float32
    Dtype for weight parameter.

Returns (forward)
-----------------
output : mx.array [B, S, NH * DH]
    Normalized and flattened multi-head activations.

[FUNCTION] __call__
------------------------------------------------------------
Apply RMSNorm using Metal kernel.

Parameters
----------
x : mx.array [..., dims]
    Input tensor (arbitrary leading dimensions).

Returns
-------
output : mx.array [..., dims]
    Normalized output matching input shape and dtype.



## Other

### `blocks/soft_cap/softcap.py`

**Bounded activation for gate stability.** One of the most important numerical tricks in this entire codebase.

**The Problem:**
Gate preactivations (input, forget, output gates) can grow arbitrarily large during training. When you exponentiate them for gating:
```
gate = exp(preactivation - stabilizer)
```
Large preactivations cause:
- Numerical overflow (exp(50) → inf)
- Saturated gradients (vanishing backprop)
- Training divergence

**The Solution - Soft Cap:**
```
soft_cap(x, c) = c * tanh(x / c)
```

This smoothly bounds preactivations to (-c, +c) without hard clipping. Properties:
- Smooth and differentiable everywhere
- Near-identity for |x| << c (doesn't interfere with small values)
- Asymptotically approaches ±c for large |x|
- Derivative: sech²(x/c) ∈ (0, 1] (always flows gradients)

**Typical cap values:**
- Gates (i, f, o): c = 15.0 (xLSTM-7B standard)
- Logits: c = 30.0 (prevents overconfident predictions)

**Metal implementation:** Element-wise kernel applying `cap * tanh(x / cap)` in parallel. Simple but critical.

### Numerical Stability Throughout

This isn't an afterthought—numerical stability is baked into every recurrent kernel:

**Log-space gating:**
```python
# BAD: Can overflow
f_gate = sigmoid(f_preact)

# GOOD: Stable log-space
f_log = -log(1 + exp(-f_preact))  # logsigmoid
f_gate = exp(f_log + stabilizer_adjustment)
```

**Stabilizer tracking:**
```python
# Update stabilizer to keep scales reasonable
m_t = max(f_log + m_{t-1}, i_t)

# Gates relative to stabilizer
f_gate = exp(f_log + m_{t-1} - m_t)
i_gate = exp(i_t - m_t)
```

**Denominator safety:**
```python
# Never divide by near-zero
denom = max(abs(q · n), exp(-m)) + eps
h = numerator / denom
```

These patterns appear in both mLSTM and sLSTM kernels. Getting them right took significant debugging against the canonical PyTorch implementation.


[MODULE] softcap
------------------------------------------------------------
Soft-Cap Function – MLX Implementation (Metal-Accelerated Bounded Activation)

Overview
--------
Soft-cap applies a smooth, bounded activation function to prevent extreme
values from destabilizing exponential operations in gating mechanisms:

  soft_cap(x, c) = c * tanh(x / c)

where c > 0 is the cap value. This function:
  - Is smooth and differentiable everywhere
  - Asymptotically approaches ±c as x → ±∞
  - Reduces to identity near zero (tanh(x/c) ≈ x/c for small x)

Why Soft-Cap?
-------------
In xLSTM, gate preactivations (input, forget, output gates) can grow large
during training. When these are exponentiated for stabilized gating:
  gate = exp(preactivation - stabilizer)
Large preactivations cause:
  1. Numerical overflow (exp(large) → inf)
  2. Gradient instability (saturated tanh derivatives → vanishing gradients)
  3. Training divergence (unstable exponential accumulation)

Soft-cap bounds preactivations to [-c, +c] smoothly, preventing these issues
while preserving gradient flow for moderate values.

Mathematical Properties
-----------------------
- Domain: ℝ → Range: (-c, +c)
- Monotonic: strictly increasing
- Odd function: soft_cap(-x, c) = -soft_cap(x, c)
- Derivative: d/dx[soft_cap(x,c)] = sech²(x/c) ∈ (0, 1]
- Near-identity: for |x| << c, soft_cap(x,c) ≈ x

Typical Cap Values
------------------
- Gates (i, f, o): c = 15.0 (xLSTM-7B default)
- Output logits: c = 30.0 (prevents overconfident predictions)

Metal Acceleration
------------------
This implementation uses a custom Metal kernel for element-wise soft-cap
on Apple Silicon. The kernel:
  1. Flattens input to 1D
  2. Applies cap * tanh(x / cap) element-wise in parallel
  3. Reshapes back to original shape

Supports float32 and bfloat16 with dtype-specific shader compilation.

Usage
-----
Functional interface (most common):
  from xlstm_metal.mlx_jit.blocks.soft_cap import soft_cap
  y = soft_cap(x, cap_value=15.0)

Module interface (for precompilation):
  cap_cell = SoftCapCell(cap_value=15.0)
  cap_cell.precompile((mx.float32, mx.bfloat16))
  y = cap_cell(x)

Gradient Flow
-------------
The derivative sech²(x/c) is always positive and bounded, ensuring stable
gradient backpropagation. Unlike hard clipping (non-differentiable at
boundaries), soft-cap maintains smooth gradients throughout.

Parity
------
Logic mirrors torch-native soft_cap for cross-backend testing.

[CLASS] SoftCapKernel
------------------------------------------------------------
Metal kernel for element-wise soft-cap with dtype caching.

Compiles and caches Metal shaders for float32/bfloat16. The kernel
implements cap * tanh(x / cap) in parallel across all elements.

Kernel Strategy
---------------
- Thread count: input size (one thread per element)
- Threadgroup size: 256 threads
- Computation: reads x[i], cap, computes y = cap * tanh(x / cap), writes y[i]

Parameters
----------
None (stateless, kernel cache is instance attribute)

Methods
-------
build(dtype) -> metal_kernel
    Compile and return cached kernel for given dtype.
precompile(dtypes)
    Pre-compile kernels for multiple dtypes at initialization.

[CLASS] SoftCapCell
------------------------------------------------------------
NCPS-style soft-cap cell with Metal kernel backend.

Applies bounded activation to prevent extreme values in gating.
Handles arbitrary input shapes via flattening/reshaping.

Parameters
----------
cap_value : float | None, optional
    Default cap value (can be overridden per call).
kernel : SoftCapKernel | None, optional
    Custom kernel instance (default creates new).

Returns (forward)
-----------------
output : mx.array
    Soft-capped activations, same shape as input.

Examples
--------
>>> cap = SoftCapCell(cap_value=15.0)
>>> x = mx.array([0.0, 10.0, 20.0, 50.0])
>>> y = cap(x)
>>> # y ≈ [0.0, 9.96, 14.76, 15.0] (bounded to ~15)

[FUNCTION] precompile
------------------------------------------------------------
:param dtypes:

[FUNCTION] __call__
------------------------------------------------------------
Apply soft-cap activation using Metal kernel.

Parameters
----------
x : mx.array
    Input tensor (arbitrary shape).
cap_value : float | mx.array | None, optional
    Cap value (uses default_cap if None).

Returns
-------
output : mx.array
    Soft-capped tensor matching input shape.

Raises
------
ValueError
    If cap_value <= 0 (cap must be positive).



### `generate.py`


[MODULE] generate
------------------------------------------------------------
Generic xLSTM Inference Runner (Config-Driven)

Provides text generation interface for any xLSTM model size by loading
configuration from the model directory or HuggingFace Hub.

[CLASS] xLSTMRunner
------------------------------------------------------------
Generic inference runner for xLSTM models (config-driven).

Automatically loads model configuration from config.json and creates
the appropriate model architecture. Works with any xLSTM model size.

Example:
    >>> # From local directory
    >>> runner = xLSTMRunner("xlstm_7b_model")

    >>> # From HuggingFace Hub
    >>> runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

    >>> # Generate text
    >>> output = runner.generate("Hello, world!", max_tokens=50)
    >>> print(output)

Comparison to PyTorch:
    PyTorch: model = AutoModelForCausalLM.from_pretrained("NX-AI/xLSTM-7b")
    MLX:     runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

[FUNCTION] __init__
------------------------------------------------------------
Initialize xLSTM runner from model directory.

Args:
    model_path: Path to model directory containing config.json and weights

The model configuration is loaded automatically from config.json,
eliminating the need for hardcoded model size parameters.

[FUNCTION] from_pretrained
------------------------------------------------------------
Load xLSTM model from HuggingFace Hub or local directory.

This method matches the PyTorch transformers API and automatically
downloads the model if needed.

Args:
    model_id: HuggingFace model ID (e.g., "NX-AI/xLSTM-7b") or local path
    cache_dir: Directory to cache downloaded models (default: ~/.cache/huggingface)
    force_download: Force re-download even if cached

Returns:
    xLSTMRunner instance with loaded model

Example:
    >>> # Download from HuggingFace Hub
    >>> runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

    >>> # Use local directory
    >>> runner = xLSTMRunner.from_pretrained("./local_model")

    >>> # Custom cache directory
    >>> runner = xLSTMRunner.from_pretrained(
    ...     "NX-AI/xLSTM-7b",
    ...     cache_dir="./my_cache"
    ... )

[FUNCTION] reset_state
------------------------------------------------------------
Reset the internal state for stateful generation.

[FUNCTION] forward
------------------------------------------------------------
Forward pass through the model.

Args:
    input_ids: Input token IDs [B, S]
    state: Optional state dict for stateful generation

Returns:
    logits: Output logits [B, S, vocab_size]
    new_state: Updated state dict

[FUNCTION] generate_next_token
------------------------------------------------------------
Generate next token given input token IDs.

Args:
    input_ids: Input token IDs [1, S]
    temperature: Sampling temperature
    top_k: Top-k sampling (None for no filtering)
    top_p: Nucleus sampling threshold (None for no filtering)

Returns:
    Next token ID as an MLX scalar array

[FUNCTION] generate
------------------------------------------------------------
Generate tokens autoregressively.

Args:
    prompt_ids: Input prompt token IDs
    max_tokens: Maximum number of tokens to generate
    temperature: Sampling temperature
    top_k: Top-k sampling
    top_p: Nucleus sampling threshold
    stop_tokens: List of token IDs to stop on

Returns:
    List of generated token IDs (including prompt)

[FUNCTION] get_model_info
------------------------------------------------------------
Get information about the model structure.

Returns:
    Dictionary with model info including config values

[FUNCTION] _prepare_prompt
------------------------------------------------------------
Normalize prompt IDs and enforce BOS/stop-token defaults.



### `tokenizer/__init__.py`


[MODULE] __init__
------------------------------------------------------------
Tokenizer blocks for MAD system



### `tokenizer/block.py`


[MODULE] block
------------------------------------------------------------
Tokenizer Block for MAD System

Wraps HuggingFace tokenizers as MAD blocks.

[CLASS] TokenizerConfig
------------------------------------------------------------
Configuration for tokenizer block.

[CLASS] TokenizerBlock
------------------------------------------------------------
Tokenizer block for MAD wiring.

This wraps a HuggingFace tokenizer and provides encode/decode methods
that work with MLX arrays.

Example:
    >>> config = TokenizerConfig(model_path="NX-AI/xLSTM-7b")
    >>> tokenizer = TokenizerBlock(config)
    >>> ids = tokenizer.encode("Hello, world!")
    >>> text = tokenizer.decode(ids)

[FUNCTION] __init__
------------------------------------------------------------
Initialize tokenizer block.

Args:
    config: TokenizerConfig with model path and settings

[FUNCTION] _load_tokenizer
------------------------------------------------------------
Load HuggingFace tokenizer (lazy).

[FUNCTION] _ensure_tokenizer
------------------------------------------------------------
Ensure tokenizer is loaded.

[FUNCTION] encode
------------------------------------------------------------
Encode text to token IDs.

Args:
    text: String or list of strings to encode

Returns:
    MLX array of token IDs [S] or [B, S]

[FUNCTION] decode
------------------------------------------------------------
Decode token IDs to text.

Args:
    ids: Token IDs as MLX array or list

Returns:
    Decoded text string or list of strings

[FUNCTION] vocab_size
------------------------------------------------------------
Get vocabulary size.

[FUNCTION] eos_token_id
------------------------------------------------------------
Get EOS token ID.

[FUNCTION] bos_token_id
------------------------------------------------------------
Get BOS token ID.

[FUNCTION] pad_token_id
------------------------------------------------------------
Get PAD token ID.



---

## Utilities

### `utils/__init__.py`


[MODULE] __init__
------------------------------------------------------------
xLSTM Inference Utilities

Configuration loading, weight loading, and checkpoint inference.



### `utils/config_loader.py`


[MODULE] config_loader
------------------------------------------------------------
Configuration Loader – MLX Implementation (HuggingFace Config Parsing)

Overview
--------
Loads xLSTM model configuration from HuggingFace-style `config.json` files
and computes derived dimensions following canonical dimension rounding rules.

This module bridges the gap between:
  - HuggingFace checkpoint format (config.json with base hyperparameters)
  - Runtime model instantiation (needs computed dimensions)

Why Separate Config Loading?
-----------------------------
Model checkpoints store **base parameters** (embedding_dim, qk_dim_factor)
but runtime code needs **computed dimensions** (qk_dim, qk_head_dim). This
loader:
  1. Reads config.json
  2. Computes derived dimensions with proper rounding
  3. Fills missing defaults for inference mode
  4. Returns a complete runtime configuration dict

Dimension Computation
---------------------
Given base parameters:
  - embedding_dim (e.g., 4096)
  - qk_dim_factor (e.g., 0.5)
  - num_heads (e.g., 8)
  - mlstm_round_up_to_multiple_of (e.g., 64)

Derived dimensions:
  qk_dim_raw = int(embedding_dim * qk_dim_factor)  # 2048
  qk_dim = round_up(qk_dim_raw, 64)                # 2048 (already multiple)
  qk_head_dim = qk_dim // num_heads                # 256

Rounding ensures dimensions align with hardware SIMD widths and safetensors
weight shapes.

Configuration Hierarchy
-----------------------
1. **Required** (must be in config.json):
   - embedding_dim, vocab_size, num_blocks, num_heads
   - qk_dim_factor, v_dim_factor, ffn_proj_factor
   - gate_soft_cap, norm_eps, use_bias

2. **Optional with defaults**:
   - chunk_size: 64
   - autocast_kernel_dtype: "bfloat16"
   - inference_state_dtype: "float32"
   - norm_reduction_force_float32: True
   - max_inference_chunksize: 16384

3. **Computed**:
   - qk_dim, v_dim, ffn_hidden_dim
   - qk_head_dim, v_head_dim

MLX vs PyTorch Config
----------------------
- **MLX**: Uses plain dicts (config['embedding_dim'])
- **PyTorch**: Uses dataclass objects (config.embedding_dim)

This module produces MLX-style dicts. For PyTorch compat, see
torch_native config loaders.

Inference Defaults
------------------
When `mode` is not specified, defaults to "inference" which:
  - Uses bfloat16 for compute (autocast_kernel_dtype)
  - Uses float32 for state (inference_state_dtype)
  - Enables return_last_states for stateful generation
  - Sets max chunk size for memory-efficient prefill

Usage
-----
Basic loading:
  >>> config = load_config("xlstm_7b_model")
  >>> model = WiredxLSTM.from_config(config)

Extract mLSTM-specific config:
  >>> mlstm_cfg = get_mlstm_config(config)
  >>> block = mLSTMBlock(**mlstm_cfg)

Parity
------
Dimension computation mirrors transformers.xLSTMConfig for checkpoint
compatibility.

[FUNCTION] _round_up
------------------------------------------------------------
Round value up to nearest multiple (matches HuggingFace xLSTMConfig).

Parameters
----------
value : int
    Raw dimension value.
multiple_of : int
    Alignment boundary (typically 64 for SIMD/safetensors).

Returns
-------
rounded : int
    Value rounded up to nearest multiple.

Examples
--------
>>> _round_up(2048, 64)
2048
>>> _round_up(2050, 64)
2112

[FUNCTION] load_config
------------------------------------------------------------
Load xLSTM configuration from HuggingFace model directory.

Reads config.json, computes derived dimensions, fills defaults.

Parameters
----------
model_path : str
    Path to model directory containing config.json (or path to config.json directly).

Returns
-------
config : dict
    Complete configuration with base + derived + default parameters.

Raises
------
FileNotFoundError
    If config.json not found in model_path.

Examples
--------
>>> config = load_config("xlstm_7b_model")
>>> config['embedding_dim']
4096
>>> config['qk_dim']  # computed from qk_dim_factor
2048
>>> config['qk_head_dim']  # computed from qk_dim / num_heads
256

[FUNCTION] get_mlstm_config
------------------------------------------------------------
Extract mLSTM block parameters from full model config.

Filters full config to only parameters needed for mLSTMBlock construction.

Parameters
----------
config : dict
    Full model configuration (from load_config).

Returns
-------
mlstm_config : dict
    Subset of config relevant to mLSTMBlock initialization.

Example
-------
>>> full_config = load_config("xlstm_7b_model")
>>> mlstm_cfg = get_mlstm_config(full_config)
>>> block = mLSTMBlock(**mlstm_cfg)

[FUNCTION] load_safetensor_shards
------------------------------------------------------------
Load every safetensor shard in ``model_path`` using ``mx.load``.

Args:
    model_path: Directory containing model.safetensors.* files
    index_filename: Name of the HuggingFace shard index (default: model.safetensors.index.json)

Returns:
    Dict mapping tensor names -> MX arrays containing the weights.

Raises:
    FileNotFoundError if the index file or shard files are missing.



### `utils/dtype_utils.py`


[MODULE] dtype_utils
------------------------------------------------------------
Dtype Utilities – MLX Implementation (String to Dtype Mapping)

Overview
--------
Provides utilities for mapping human-readable dtype strings (from config files)
to MLX dtype objects. This enables configuration-driven precision selection
without hardcoding dtype objects.

Why String-Based Dtype Config?
-------------------------------
Configuration files (JSON, YAML) cannot directly encode programming objects.
By using string identifiers like "float32", "bfloat16", configs remain:
  - Human-readable and editable
  - Framework-agnostic (same strings work for MLX, PyTorch, JAX)
  - Version-stable (no serialization dependencies)

Supported Dtypes
----------------
- "float32" → mx.float32 (standard 32-bit floating point)
- "float16" → mx.float16 (half precision, 16-bit)
- "bfloat16" or "bf16" → mx.bfloat16 (brain float, 16-bit with float32 range)
- "fp16" → mx.float16 (alias for float16)

Mixed Precision Patterns
------------------------
Common configurations in xLSTM:
  - compute_dtype: "float32" or "bfloat16" (forward pass activations)
  - state_dtype: "float32" (recurrent state for precision)
  - param_dtype: "float32" (trainable parameters)

Using bfloat16 for compute while keeping state in float32 reduces memory
while preserving numerical stability in long-range accumulations.

BFloat16 vs Float16
-------------------
- **Float16**: Standard IEEE half precision
  - Exponent: 5 bits, Mantissa: 10 bits
  - Range: ~6e-8 to 65504
  - Issues: Limited range can cause overflow/underflow

- **BFloat16**: Brain float (Google TPU format)
  - Exponent: 8 bits (same as float32), Mantissa: 7 bits
  - Range: ~1e-38 to 3e38 (same as float32)
  - Advantages: Wider range, simpler float32 ↔ bf16 conversion

For LLM training, bfloat16 is typically preferred over float16 due to
better numerical stability with large models.

Usage
-----
In config.json:
  {
    "autocast_kernel_dtype": "bfloat16",
    "inference_state_dtype": "float32"
  }

In Python:
  from xlstm_metal.mlx_jit.utils import resolve_dtype
  compute_dtype = resolve_dtype(config['autocast_kernel_dtype'])
  state_dtype = resolve_dtype(config['inference_state_dtype'])

Fallback Behavior
-----------------
If an unknown string is provided, `resolve_dtype` falls back to the
default dtype (float32) to prevent crashes. This enables forward
compatibility with new dtype strings.

Parity
------
Logic mirrors torch-native dtype_utils for cross-backend compatibility.

[FUNCTION] resolve_dtype
------------------------------------------------------------
Map config dtype string to MLX dtype object.

Parameters
----------
name : str | None
    Dtype string from config ("float32", "bfloat16", "bf16", etc.).
    If None, uses default.
default : str, default "float32"
    Fallback dtype string if name is None or unrecognized.

Returns
-------
dtype : mx.Dtype
    Corresponding MLX dtype object.

Examples
--------
>>> resolve_dtype("bfloat16")
mlx.core.bfloat16
>>> resolve_dtype(None)  # uses default
mlx.core.float32
>>> resolve_dtype("unknown_dtype")  # fallback to default
mlx.core.float32



### `utils/gguf_loader.py`


[MODULE] gguf_loader
------------------------------------------------------------
GGUF checkpoint loader and config inference.

Parses GGUF format to extract model configuration and weights.

[CLASS] GGUFReader
------------------------------------------------------------
Basic GGUF file reader.

GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

[FUNCTION] infer_config_from_gguf
------------------------------------------------------------
Infer model configuration from GGUF checkpoint.

Args:
    model_path: Path to GGUF model file

Returns:
    Dict with inferred configuration

Raises:
    NotImplementedError: GGUF support not yet implemented

[FUNCTION] read_header
------------------------------------------------------------
Read GGUF header and metadata.

[FUNCTION] _read_kv_pair
------------------------------------------------------------
Read key-value pair from GGUF.

[FUNCTION] _read_tensor_info
------------------------------------------------------------
Read tensor info from GGUF.

[FUNCTION] get_tensor_shape
------------------------------------------------------------
Get shape of a tensor.



### `utils/infer_config_from_checkpoint.py`


[MODULE] infer_config_from_checkpoint
------------------------------------------------------------
Checkpoint-agnostic config inference dispatcher.

Tries to infer config from checkpoint (safetensors/GGUF) first,
falls back to config.json with warning.

[FUNCTION] infer_config_from_checkpoint
------------------------------------------------------------
Infer model configuration from checkpoint, with fallbacks.

Priority order:
1. Safetensors (if model.safetensors.index.json exists)
2. GGUF (if .gguf file exists)
3. config.json (with warning about non-model-agnostic loading)

Args:
    model_path: Path to model directory

Returns:
    Dict with model configuration

Raises:
    FileNotFoundError: If no valid checkpoint or config found



### `utils/infer_config_from_safetensors.py`


[MODULE] infer_config_from_safetensors
------------------------------------------------------------
Config inference from safetensors checkpoint.

Derives model architecture parameters directly from tensor shapes in the checkpoint,
making the model loading fully model-agnostic.

[FUNCTION] _shape_of
------------------------------------------------------------
Get shape of a tensor from index metadata or loaded shards.

Args:
    tensor_name: Name of the tensor
    index: The safetensors index dict
    shards: Dict of loaded shard arrays

Returns:
    Shape tuple or None if not found

[FUNCTION] _parse_block_index
------------------------------------------------------------
Parse a single numeric string component (e.g., the '2' from 'backbone.blocks.2')
and convert it to an integer without direct casts.

[FUNCTION] _infer_heads_from_mhln
------------------------------------------------------------
Infer number of heads from multihead_norm weight shape.

Args:
    model_dir: Path to model directory
    d_model: Model embedding dimension

Returns:
    Number of heads

[FUNCTION] _infer_gate_soft_cap_default
------------------------------------------------------------
Default gate soft cap value.

[FUNCTION] _infer_norm_eps_default
------------------------------------------------------------
Default normalization epsilon.

[FUNCTION] infer_config_from_safetensors
------------------------------------------------------------
Infer complete model configuration from safetensors checkpoint.

Reads tensor shapes to derive all architectural parameters, making
the model loading fully checkpoint-driven and model-agnostic.

Args:
    model_dir: Path to model directory with safetensors

Returns:
    Dict with inferred configuration

Raises:
    FileNotFoundError: If safetensors index not found
    ValueError: If required tensors missing or shapes inconsistent

[FUNCTION] get_shape
------------------------------------------------------------
:param name:
:return:



### `utils/safetensors_loader.py`


[MODULE] safetensors_loader
------------------------------------------------------------
Direct safetensors weight loader for xLSTM-7B MAD model.

Loads HuggingFace safetensors directly into WiredMADModel without conversion.
Uses the canonical match_dict from xlstm-jax.

[FUNCTION] load_safetensors_into_wired_model
------------------------------------------------------------
Load xLSTM-7B weights directly from HuggingFace safetensors into the NCPS Wired model.

Based on xlstm-jax canonical implementation:
xlstm_jax/utils/model_param_handling/handle_mlstm_simple.py

Args:
    model_dir: Path to HuggingFace model directory with safetensors
    model: WiredxLSTM instance to load weights into



### `utils/weight_loader.py`


[MODULE] weight_loader
------------------------------------------------------------
Weight loader for xLSTM-7B NPZ weights → MAD blocks

Maps the flattened NPZ weight structure to MAD block hierarchy.
Supports both standalone xLSTMBlock lists and WiredMADModel instances.

[FUNCTION] load_npz_weights_to_block
------------------------------------------------------------
Load weights from NPZ into an xLSTMBlock.

NPZ structure (from convert_hf_to_mlx.py):
    blocks.{i}.W_q.weight -> xlstm.q.weight
    blocks.{i}.W_k.weight -> xlstm.k.weight
    blocks.{i}.W_v.weight -> xlstm.v.weight
    blocks.{i}.W_i.weight/bias -> xlstm.igate_preact.weight/bias
    blocks.{i}.W_f.weight/bias -> xlstm.fgate_preact.weight/bias
    blocks.{i}.W_o.weight -> xlstm.out_proj.weight
    blocks.{i}.mhln.weight -> xlstm.multihead_norm.weight
    blocks.{i}.norm.weight -> xlstm_norm.weight
    blocks.{i}.norm2.weight -> ffn_norm.weight
    blocks.{i}.up_l_proj.weight -> ffn.proj_up (gate part)
    blocks.{i}.up_r_proj.weight -> ffn.proj_up (up part)
    blocks.{i}.down_proj.weight -> ffn.proj_down.weight

Args:
    npz_weights: Loaded NPZ weights dict
    block_idx: Block index (0-31 for xLSTM-7B)
    block: xLSTMBlock to load weights into

Note: The NPZ uses `up_l_proj` and `up_r_proj` for gate+up projections,
      but our GatedFFN uses a single `proj_up` that outputs 2x dims.
      We need to concatenate them.

[FUNCTION] load_xLSTM_7b_weights
------------------------------------------------------------
Load xLSTM-7B weights from NPZ file into list of xLSTMBlocks.

Args:
    npz_path: Path to xlstm_7b_mlx_converted.npz
    blocks: List of 32 xLSTMBlock instances

Returns:
    embedding_weight: [vocab_size, embedding_dim]
    head_weight: [vocab_size, embedding_dim]

[FUNCTION] load_weights_into_wired_model
------------------------------------------------------------
Load xLSTM-7B weights from NPZ file into a WiredxLSTM model.

This function handles the MAD wiring structure where blocks are named
'xlstm_0', 'xlstm_1', etc., and also loads embedding and lm_head weights.

Args:
    npz_path: Path to xlstm_7b_mlx_converted.npz
    model: WiredxLSTM instance with xLSTM-7B wiring

The model is expected to have blocks named:
    - 'embedding': Token embedding
    - 'xlstm_0' through 'xlstm_31': xLSTM blocks
    - 'final_norm': Final RMSNorm
    - 'lm_head': Language model head



---

## mLSTM Backward Kernels

### `blocks/mlstm/mlstm_chunkwise/backward/mlstm_chunkwise_parallel_bw_dK.py`


[MODULE] mlstm_chunkwise_parallel_bw_dK
------------------------------------------------------------
Metal kernel for parallel part of backward pass computing dK gradients.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes ∂Loss/∂K using intra-chunk and inter-chunk contributions.

[FUNCTION] mlstm_chunkwise_parallel_bw_dK_metal
------------------------------------------------------------
Metal kernel for parallel backward computation of dK gradients.

Returns:
    matDeltaK: (B, NH, S, DHQK) - gradients w.r.t. K



### `blocks/mlstm/mlstm_chunkwise/backward/mlstm_chunkwise_parallel_bw_dQ.py`


[MODULE] mlstm_chunkwise_parallel_bw_dQ
------------------------------------------------------------
Metal kernel for parallel part of backward pass computing dQ gradients.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes ∂Loss/∂Q using intra-chunk and inter-chunk contributions.

[FUNCTION] mlstm_chunkwise_parallel_bw_dQ_metal
------------------------------------------------------------
Metal kernel for parallel backward computation of dQ gradients.

Returns:
    matDeltaQ: (B, NH, S, DHQK) - gradients w.r.t. Q



### `blocks/mlstm/mlstm_chunkwise/backward/mlstm_chunkwise_parallel_bw_dV.py`


[MODULE] mlstm_chunkwise_parallel_bw_dV
------------------------------------------------------------
Metal kernel for parallel part of backward pass computing dV gradients.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes ∂Loss/∂V using intra-chunk and inter-chunk contributions.

[FUNCTION] mlstm_chunkwise_parallel_bw_dV_metal
------------------------------------------------------------
Metal kernel for parallel backward computation of dV gradients.

Returns:
    matDeltaV: (B, NH, S, DHHV) - gradients w.r.t. V



### `blocks/mlstm/mlstm_chunkwise/backward/mlstm_chunkwise_recurrent_bw_dC.py`


[MODULE] mlstm_chunkwise_recurrent_bw_dC
------------------------------------------------------------
Metal kernel for recurrent part of backward pass of the mLSTM chunkwise formulation.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes gradient deltas for C states by backpropagating through chunks
in reverse order (last to first).

[FUNCTION] mlstm_chunkwise_recurrent_bw_dC_metal
------------------------------------------------------------
Metal kernel for backward recurrent computation of mLSTM gradient deltas.

Backpropagates through chunks in reverse order (last to first).

Returns:
    matDeltaC_states: (B, NH, (NC+1)*DHQK, DHHV) - gradient deltas for C states
    :param matQ:
    :param vecF:
    :param scaM_inter:
    :param vecM_combine:
    :param matDeltaH:
    :param vecN_out:
    :param matDeltaC_last:
    :param NC:
    :param L:
    :param qk_scale:
    :param siz_b_DHQK:
    :param siz_b_DHHV:
    :param save_states_every_nth_chunk:
    :param eps:
    :return:



---

## mLSTM Block

### `blocks/mlstm/__init__.py`


[MODULE] __init__
------------------------------------------------------------
mLSTM MLX Implementation for xLSTM-7B

MLX-based mLSTM blocks (high-level operations, runs on CPU/Metal/CUDA).



### `blocks/mlstm/mlstm_block.py`


[MODULE] mlstm_block
------------------------------------------------------------
mLSTM Block (xLSTM-7B) – MLX Implementation

Overview
--------
The mLSTM block is the *matrix memory* component of xLSTM. Each attention
head maintains matrix-valued state C plus accompanying normalizers (n, m)
allowing long-range sequence modeling with stabilized exponential updates.

Composition
-----------
This high-level block wraps three conceptual cells:
  1. Projection Cell (before): input -> Q,K,V + gate preactivations
  2. Kernel Cell     (during): chunkwise or recurrent memory updates
  3. Output Cell     (after) : per-head normalization + gating + projection

The block then appends a SwiGLU feed-forward network (FFN) with its own
pre-normalization (RMSNorm). Two residual connections are applied:
  (a) input + mLSTM output
  (b) intermediate + FFN output

Sequence of Operations
----------------------
residual = x
x_normed = norm_mlstm(x)
mlstm_out, state = mlstm_cell(x_normed, state)
x = residual + mlstm_out

residual = x
x_normed = norm_ffn(x)
ffn_hidden = silu(proj_up_gate(x_normed)) * proj_up(x_normed)
ffn_out    = proj_down(ffn_hidden)
x = residual + ffn_out

State Structure (returned by mlstm_cell)
---------------------------------------
state = (C, n, m)
  C : [B, NH, DH_qk, DH_v]  memory matrices per head
  n : [B, NH, DH_qk]        normalizer vector per head
  m : [B, NH]               stabilized scalar log-sum per head

Dimension Rounding
------------------
To match pretrained safetensors, QK and V per-head dims are rounded up to
`mlstm_round_up_to_multiple_of`, and FFN hidden dim to `ffn_round_up_to_multiple_of`.

SwiGLU FFN
----------
Applies gate = silu(W_gate x) and up = W_up x then elementwise gate * up
followed by a down projection. This is standard for modern transformer blocks.

Numeric Stability
-----------------
- Norm reductions may force float32 (`norm_reduction_force_float32`) to
  reduce precision loss for mixed dtype kernels.
- `eps` ensures denominator safety in normalization and gating steps.

Metal Kernels
-------------
`kernel_mode` selects specialized MLX Metal kernels for parallel chunkwise
execution (`metal_chunkwise`) or fallback recurrent paths.

Parity & Torch Backend
----------------------
Logic mirrors the torch_native `mLSTMBlock` to enable forward parity tests
across backends.

[CLASS] mLSTMBlock
------------------------------------------------------------
High-level mLSTM + FFN block (matrix memory + SwiGLU) with residuals.

Wraps projection, kernel, and output processing for mLSTM plus the
feed-forward network. Provides weight key mapping for safetensors
loading and configuration introspection.

Parameters
----------
block_index : int
    Index of the block inside the backbone (0-based).
embedding_dim : int, default 4096
    Model embedding dimension.
num_heads : int, default 8
    Number of attention heads.
qk_dim_factor : float, default 0.5
    Proportion of embedding_dim allocated to Q/K per head (pre-rounding).
v_dim_factor : float, default 1.0
    Proportion for V per head (commonly 1.0 in xLSTM).
gate_soft_cap : float, default 15.0
    Soft cap applied to gate preactivations inside cells.
ffn_proj_factor : float, default 2.667
    Expansion ratio for FFN hidden dimension before rounding.
ffn_round_up_to_multiple_of : int, default 64
    Alignment multiple for FFN hidden dimension.
mlstm_round_up_to_multiple_of : int, default 64
    Alignment multiple for QK / V per-head dims.
chunk_size : int, default 64
    Chunk size for parallel kernel execution.
kernel_mode : str, default "metal_chunkwise"
    Execution strategy/preset for kernels.
norm_eps : float, default 1e-6
    Epsilon for RMSNorm stability.
norm_reduction_force_float32 : bool, default True
    Force float32 accumulation in norm reductions.
use_bias : bool, default False
    Whether linear layers use bias (weights are typically no-bias).
eps : float, default 1e-6
    Numeric epsilon for internal gating/normalization.
sparsity_mask : Optional[mx.array]
    Optional block-level sparsity wiring mask.
compute_dtype : mx.Dtype, default mx.float32
    Dtype for forward activations.
state_dtype : mx.Dtype, default mx.float32
    Dtype for recurrent state tensors (can differ for memory footprint).

Returns (forward)
-----------------
output : mx.array [B, S, embedding_dim]
new_state : (C, n, m) recurrent state tuple.

Notes
-----
- Residual connections use pre-norm design.
- Weight tying / LM head applied outside this block.
- Compatible with automatic wiring / NCPS model assembly.

[FUNCTION] _round_up_to_multiple
------------------------------------------------------------
Round up value to nearest multiple.

[FUNCTION] __call__
------------------------------------------------------------
Forward pass through composite mLSTM block.

Parameters
----------
x : mx.array [B, S, embedding_dim]
    Input activations for this block.
state : tuple | None
    Previous mLSTM state (C, n, m) or None to initialize internally.

Returns
-------
output : mx.array [B, S, embedding_dim]
    Activation after mLSTM + FFN + residual pathways.
new_state : (C, n, m)
    Updated recurrent state from mLSTM kernel cell.

[FUNCTION] get_weight_keys
------------------------------------------------------------
Get mapping of module parameters to safetensors keys.

Returns:
    Dict mapping self.parameter_path -> safetensors_key
    
Example:
    {
        "norm_mlstm.weight": "backbone.blocks.0.norm_mlstm.weight",
        "mlstm_cell.q_proj.weight": "backbone.blocks.0.mlstm_layer.q.weight",
        ...
    }

[FUNCTION] from_config
------------------------------------------------------------
Create cell from config.json dict.

Args:
    block_index: Block index (0-31 for xLSTM-7B)
    config: Config dict from config.json
    sparsity_mask: Optional NCPS wiring sparsity mask
    
Returns:
    Initialized mLSTMBlock
    
Example:
    >>> import json
    >>> with open("xlstm_7b_model/config.json") as f:
    ...     config = json.load(f)
    >>> cell = mLSTMBlock.from_config(0, config)

[FUNCTION] get_config
------------------------------------------------------------
Return configuration for serialization.



---

## mLSTM Cells

### `blocks/mlstm/mlstm_chunkwise/__init__.py`


[MODULE] __init__
------------------------------------------------------------
mLSTM Chunkwise Implementation - Clean NCPS Architecture.

Modular mLSTM with clean separation of concerns:
- Projection Cell: Input transformations (Q/K/V, gates)
- Kernel Cells: Pure recurrence logic (parallel vs recurrent)
- Output Cell: Output transformations (norm, gate, projection)
- Neuron: Wires cells together with dispatch logic



### `blocks/mlstm/mlstm_chunkwise/mlstm_neuron.py`


[MODULE] mlstm_neuron
------------------------------------------------------------
mLSTM Neuron – MLX Implementation (Projection + Kernel + Output)

Overview
--------
The mLSTM neuron is the *per-block* composite that wires the three phases
of matrix-memory processing:
  1. Projection Cell (before): input → Q, K, V + gate preactivations
  2. Kernel Cell     (during): recurrence (parallel chunkwise or sequential) producing hidden states h
  3. Output Cell     (after) : per-head normalization, output gating, final projection back to input size

It owns the dispatch decision between chunkwise parallel kernels and
sequential recurrent kernels (reduced memory path) via `kernel_mode`.

Chunkwise vs Recurrent
----------------------
- parallel  : Uses optimized Metal kernels to process sequence in fixed-size chunks for higher throughput.
- recurrent : Processes one timestep at a time; suitable for autoregressive generation and reduced memory usage.

State Structure
---------------
State tuple = (C, n, m):
  C : [B, NH, DH_qk, DH_v]  matrix-memory outer-product accumulator
  n : [B, NH, DH_qk]        normalizer accumulator for C
  m : [B, NH]               stabilized scalar gating accumulator

Data Flow (forward)
-------------------
Input x [B, S, D]
  → projection_cell → q,k,v (reshaped) + i_preact,f_preact gates
  → kernel_cell     → hidden h [B, NH, S, DH_v], new_state (C,n,m)
  → output_cell     → output [B, S, D]

Numerical Stability
-------------------
The kernel cells internally apply stabilized exponentials (using log-space
techniques similar to sLSTM) and per-head scaling to keep activations within
well-conditioned ranges for long sequences.

Dtype Handling
--------------
Projection outputs are coerced to `compute_dtype` while recurrence state is
stored in `state_dtype`. This allows mixed precision inference while retaining
higher precision in recurrent accumulators.

Parity
------
Logic mirrors torch-native `mLSTMNeuron` for cross-backend parity testing.

[CLASS] mLSTMNeuron
------------------------------------------------------------
Composite mLSTM layer orchestrating projection, kernel, and output phases.

Parameters
----------
input_size : int
    Embedding / model dimension D.
num_heads : int
    Number of attention heads (NH).
qk_dim_per_head : int
    Query/key dimension per head.
v_dim_per_head : int
    Value dimension per head.
chunk_size : int, default 64
    Sequence chunk length for parallel kernels.
kernel_mode : {"parallel", "recurrent"}, default "parallel"
    Execution mode selecting chunkwise vectorized kernels or stepwise recurrence.
use_bias : bool, default False
    Whether projection/output linear layers include bias.
eps : float, default 1e-6
    Numerical stability constant for internal operations.
gate_soft_cap : float | None, optional
    Soft-cap value for gate preactivations (None disables capping).
compute_dtype : mx.Dtype, default mx.float32
    Dtype for activations and arithmetic inside kernels.
state_dtype : mx.Dtype, default mx.float32
    Dtype for recurrent state tensors (C, n, m).
force_float32_reductions : bool, default True
    Force float32 in reduction ops (norms, sums) for stability.

Returns (forward)
-----------------
output : mx.array [B, S, D]
    Final dense representation after output processing.
new_state : (C, n, m)
    Updated recurrent memory state tuple for next forward call.

[FUNCTION] state_size
------------------------------------------------------------
Return state dimensions (C, n, m).

[FUNCTION] output_size
------------------------------------------------------------
Return output dimension.

[FUNCTION] __call__
------------------------------------------------------------
Run the full mLSTM pipeline for a sequence batch.

Parameters
----------
x : mx.array [B, S, D]
    Input embedding sequence.
state : tuple | None
    Previous recurrent state (C, n, m) or None for initialization.

Returns
-------
output : mx.array [B, S, D]
    Transformed output after memory integration and gating.
new_state : (C, n, m)
    Updated memory state for next call.

[FUNCTION] get_config
------------------------------------------------------------
Return configuration for serialization.



### `blocks/mlstm/mlstm_chunkwise/mlstm_output_cell.py`


[MODULE] mlstm_output_cell
------------------------------------------------------------
mLSTM Output Cell – MLX Implementation (After Phase)

Overview
--------
The output cell is the **"after"** component in the modular mLSTM pipeline.
It receives hidden states h from the kernel cell and the original input x,
then produces the final block output via:
  1. Per-head RMS normalization of h (flattened across heads)
  2. Output gating conditioned on the original input x
  3. Final linear projection back to input_size

It contains **no recurrence**—purely feedforward post-processing.

Pipeline Position
-----------------
Input x [B, S, D]
  → Projection Cell → (q, k, v, i_preact, f_preact)
  → Kernel Cell     → hidden h [B, NH, S, DH_v]
  → Output Cell     → output [B, S, D]

Tensor Shapes
-------------
Inputs:
  h      : [B, NH, S, DH_v]   hidden states from kernel cell
  x_orig : [B, S, D]          original input (for output gate conditioning)

Output:
  output : [B, S, D]          final processed representation

Per-Head RMS Normalization
---------------------------
RMS norm is applied per head over the head_dim axis, then results are
flattened to [B, S, NH * DH_v]. This preserves per-head statistics while
enabling a shared weight vector across all heads.

Output Gate (o_gate)
--------------------
The output gate is computed from the **original input** x_orig (before
projection or recurrence), not from the hidden states. This allows the
model to conditionally attenuate or amplify memory-derived features based
on the current input context:
  o_gate = sigmoid(W_o @ x_orig)
  h_gated = h_norm ⊙ o_gate
This is analogous to the output gate in standard LSTM but conditioned on
the embedding rather than recurrent state.

Why Use x_orig?
---------------
Using the original input for the output gate provides a **skip connection**
pattern where the model can selectively bypass the recurrent memory updates
if the input context suggests they're not relevant. This improves gradient
flow and allows more flexible gating.

Final Projection
----------------
After gating, a linear layer projects [B, S, NH * DH_v] → [B, S, D], matching
the input dimensionality for residual addition at the block level.

Force Float32 Reductions
------------------------
The norm cell may internally cast to float32 during mean/variance computation
(controlled by `force_float32_reductions`) to avoid precision loss in
mixed-precision settings (e.g., bfloat16 inference).

NCPS Terminology
----------------
In NCPS / liquid time-constant networks:
  - Output gate is the "motor neuron gate" (final control signal)
  - Normalization stabilizes "inter-layer dynamics"
This cell follows that modular, composable pattern.

Parity
------
Logic mirrors torch-native `mLSTMOutputCell` for cross-backend testing.

[CLASS] mLSTMOutputCell
------------------------------------------------------------
Output post-processing stage for mLSTM (no recurrence, just transformation).

Parameters
----------
input_size : int
    Embedding / model dimension D.
num_heads : int
    Number of attention heads (NH).
v_dim_per_head : int
    Value dimension per head.
use_bias : bool, default False
    Whether output gate and projection layers include bias.
eps : float, default 1e-6
    Epsilon for RMS normalization stability.
force_float32_reductions : bool, default True
    Force float32 in norm reductions for numerical stability.
param_dtype : mx.Dtype, default mx.float32
    Dtype for norm parameters (weight).

Returns (forward)
-----------------
output : mx.array [B, S, D]
    Final output after normalization, gating, and projection.

[FUNCTION] __call__
------------------------------------------------------------
Transform kernel hidden states to final output.

Parameters
----------
h : mx.array [B, NH, S, DH_v]
    Hidden states from the kernel cell (memory-integrated features).
x_orig : mx.array [B, S, D]
    Original input embedding (used for output gate conditioning).

Returns
-------
output : mx.array [B, S, D]
    Final output ready for residual addition at block level.



### `blocks/mlstm/mlstm_chunkwise/mlstm_parallel_kernel_cell.py`


[MODULE] mlstm_parallel_kernel_cell
------------------------------------------------------------
mLSTM Parallel Kernel Cell – MLX Implementation (Chunkwise Recurrence)

Overview
--------
The parallel kernel cell is the **"during"** (recurrence) component of
the mLSTM pipeline using a **chunkwise parallel** strategy. It processes
sequences in fixed-size chunks, computing:
  1. **Inter-chunk** recurrence (sequential across chunks)
  2. **Intra-chunk** attention (parallel within each chunk)

This approach balances parallelism (for throughput) with recurrent memory
(for long-range dependencies).

Two-Phase Algorithm
-------------------
Phase 1 (Sequential across chunks):
  For each chunk k = 0..NC-1:
    Compute inter-chunk states (C_k, n_k, m_k) from prior state + chunk content.
  This produces NC + 1 boundary states (including initial state).

Phase 2 (Parallel within chunks):
  For all positions i in all chunks (fully parallel):
    Compute hidden state h_i using:
      - Intra-chunk attention (causal within chunk)
      - Inter-chunk contribution from boundary state C_{k-1}
  Uses Metal-optimized kernel for high throughput.

Why Chunkwise?
--------------
- Full parallel attention over long sequences: O(S²) memory + compute
- Full recurrent: O(S) memory but sequential (slow for long S)
- Chunkwise: O(S · L) intra-chunk + O(S/L) inter-chunk (practical tradeoff)

Chunk Size L
------------
Typical values: 64, 128. Larger L increases intra-chunk parallelism but
requires more memory. Smaller L reduces per-chunk cost but increases
sequential overhead from inter-chunk updates.

State Structure
---------------
State tuple = (C, n, m):
  C : [B, NH, DH_qk, DH_v]  matrix-memory (outer product accumulator)
  n : [B, NH, DH_qk]        normalizer vector
  m : [B, NH]               scalar stabilizer (log-space gating)

Padding
-------
If sequence length S is not divisible by chunk_size L, the inputs are
zero-padded to NC * L (NC = ceil(S / L)). Output is then unpadded back
to length S.

Metal Kernels
-------------
Both recurrent (inter-chunk) and parallel (intra-chunk) phases call
Metal-optimized kernels for efficient execution on Apple Silicon.

Numerical Stability
-------------------
- Input/forget gate preactivations are processed in log-space to avoid
  exponential overflow.
- Query scaling (1 / sqrt(DH_qk)) applied before attention-like operations.
- Mixed precision: compute in `compute_dtype`, store state in `state_dtype`.

Parity
------
Logic mirrors torch-native `mLSTMParallelKernelCell` for cross-backend testing.

[CLASS] mLSTMParallelKernelCell
------------------------------------------------------------
Chunkwise parallel recurrence kernel for mLSTM (no projections, pure memory).

Parameters
----------
num_heads : int
    Number of attention heads (NH).
qk_dim_per_head : int
    Query/key dimension per head.
v_dim_per_head : int
    Value dimension per head.
chunk_size : int, default 64
    Chunk length L for parallel processing.
eps : float, default 1e-6
    Numerical stability epsilon.
compute_dtype : mx.Dtype, default mx.float32
    Dtype for forward pass activations.
state_dtype : mx.Dtype, default mx.float32
    Dtype for recurrent state storage (C, n, m).

Returns (forward)
-----------------
h : mx.array [B, NH, S, DH_v]
    Hidden states for all timesteps.
new_state : (C, n, m)
    Updated boundary state for next call.

[FUNCTION] __call__
------------------------------------------------------------
Execute two-phase chunkwise parallel mLSTM recurrence.

Parameters
----------
q : mx.array [B, NH, S, DH_qk]
    Query projections.
k : mx.array [B, NH, S, DH_qk]
    Key projections.
v : mx.array [B, NH, S, DH_v]
    Value projections.
i_preact : mx.array [B, NH, S]
    Input gate preactivations.
f_preact : mx.array [B, NH, S]
    Forget gate preactivations.
state : tuple | None
    Previous recurrent state (C, n, m) or None for initialization.

Returns
-------
h : mx.array [B, NH, S, DH_v]
    Hidden states computed via chunkwise algorithm.
new_state : (C, n, m)
    Final recurrent state (boundary of last chunk).



### `blocks/mlstm/mlstm_chunkwise/mlstm_projection_cell.py`


[MODULE] mlstm_projection_cell
------------------------------------------------------------
mLSTM Projection Cell – MLX Implementation (Before Phase)

Overview
--------
The projection cell is the **"before"** component in the modular mLSTM
pipeline. It receives raw input embeddings and produces:
  1. Query (Q), Key (K), Value (V) tensors reshaped for multi-head processing
  2. Input gate (i) and forget gate (f) preactivations (pre-sigmoid/softplus)

It contains **no recurrence** and **no output gating**—purely feedforward
transformations preparing inputs for the kernel cell.

Pipeline Position
-----------------
Input [B, S, D]
  → Projection Cell → (q, k, v, i_preact, f_preact)
  → Kernel Cell     → hidden states h
  → Output Cell     → final output [B, S, D]

Tensor Shapes
-------------
Inputs:
  x : [B, S, input_size]

Outputs:
  q        : [B, NH, S, DH_qk]   query (multi-head reshaped)
  k        : [B, NH, S, DH_qk]   key   (multi-head reshaped)
  v        : [B, NH, S, DH_v]    value (multi-head reshaped)
  i_preact : [B, NH, S]          input gate preactivation
  f_preact : [B, NH, S]          forget gate preactivation

Gate Preactivations
-------------------
Gates are projected to [B, S, NH] then transposed to [B, NH, S] for
per-head processing. Optional soft-cap (cap * tanh(x / cap)) bounds
magnitudes before exponential operations in the kernel cell.

Why Separate i/f Gates?
------------------------
Input and forget gates control the contribution of new vs old memory:
  - i_preact → sigmoid → weight for new content (k ⊗ v)
  - f_preact → sigmoid → weight for prior state C_{t-1}
Exponential stabilization (log-space) happens in the kernel cell.

Use Bias?
---------
Q/K/V projections typically do **not** use bias (canonical xLSTM setting).
Gate projections (igate_proj, fgate_proj) **do** use bias (initialized to
reasonable defaults) to allow learned baseline gating behavior.

Soft-Cap
--------
If `gate_soft_cap` is set, applies cap * tanh(preact / cap) to i_preact
and f_preact. This prevents extreme values that could destabilize the
exponential gating in the kernel phase.

NCPS Terminology
----------------
In NCPS / liquid time-constant networks:
  - Q/K/V projections are "feature groups" (different representational subspaces)
  - i/f gates are "excitatory/inhibitory" control signals
This projection cell follows that modular pattern.

Parity
------
Logic mirrors torch-native `mLSTMProjectionCell` for cross-backend testing.

[CLASS] mLSTMProjectionCell
------------------------------------------------------------
Input projection stage for mLSTM (no recurrence, no output processing).

Parameters
----------
input_size : int
    Embedding / model dimension D.
num_heads : int
    Number of attention heads (NH).
qk_dim_per_head : int
    Query/key dimension per head.
v_dim_per_head : int
    Value dimension per head.
use_bias : bool, default False
    Whether Q/K/V linear layers include bias (typically False).
gate_soft_cap : float | None, optional
    Soft-cap value for gate preactivations (None disables).
soft_cap_cell : SoftCapCell | None, optional
    Custom soft-cap cell instance (default creates new SoftCapCell).

Returns (forward)
-----------------
q : mx.array [B, NH, S, DH_qk]
k : mx.array [B, NH, S, DH_qk]
v : mx.array [B, NH, S, DH_v]
i_preact : mx.array [B, NH, S]
f_preact : mx.array [B, NH, S]

[FUNCTION] __call__
------------------------------------------------------------
Project input to multi-head Q/K/V and gate preactivations.

Parameters
----------
x : mx.array [B, S, input_size]
    Input embedding sequence.

Returns
-------
q : mx.array [B, NH, S, DH_qk]
    Query tensor for attention-like memory indexing.
k : mx.array [B, NH, S, DH_qk]
    Key tensor for memory content addressing.
v : mx.array [B, NH, S, DH_v]
    Value tensor for memory payload.
i_preact : mx.array [B, NH, S]
    Input gate preactivation (before sigmoid).
f_preact : mx.array [B, NH, S]
    Forget gate preactivation (before sigmoid).



### `blocks/mlstm/mlstm_chunkwise/mlstm_recurrent_kernel_cell.py`


[MODULE] mlstm_recurrent_kernel_cell
------------------------------------------------------------
mLSTM Recurrent Kernel Cell – MLX Implementation (Sequential Recurrence)

Overview
--------
The recurrent kernel cell is the **"during"** (recurrence) component of
the mLSTM pipeline using a **step-by-step sequential** strategy. It processes
each timestep one-at-a-time in a loop, maintaining and updating the
recurrent state (C, n, m) at each step.

This mode is primarily used for:
  - Autoregressive generation (one token at a time)
  - Inference with strict memory constraints
  - Debugging / validating chunkwise parallel implementations

Sequential vs Parallel
----------------------
- **Sequential (this cell)**: Processes S timesteps in a loop. O(S) memory,
  O(S · DH_qk · DH_v) compute. No parallelism across time.

- **Parallel (chunkwise cell)**: Processes chunks in parallel. O(S · L) memory,
  O(S · L + S/L · DH_qk · DH_v) compute. High throughput for training/prefill.

When to Use Sequential
-----------------------
- **Generation**: After initial prompt processing, generate one token at a
  time. Sequential mode uses constant memory per step.
- **Low memory**: When batch size * sequence length * hidden dims exceeds
  available memory.
- **Debugging**: Sequential loop is easier to trace and validate.

State Update Equations
----------------------
For each timestep t = 0..S-1:
  1. Stabilized gating:
       f_log = -log(1 + exp(-f_t))   # log(sigmoid(f_t))
       m_t = max(f_log + m_{t-1}, i_t)
       f_gate = exp(f_log + m_{t-1} - m_t)
       i_gate = exp(i_t - m_t)

  2. State updates (per head, matrix-valued):
       C_t = f_gate * C_{t-1} + i_gate * (k_t ⊗ v_t)
       n_t = f_gate * n_{t-1} + i_gate * k_t
       m_t already computed above

  3. Output computation:
       q_scaled = q_t / sqrt(DH_qk)
       h_num = sum_over_qk( C_t * q_scaled )
       h_den = max(|q_scaled · n_t|, exp(-m_t)) + eps
       h_t = h_num / h_den

State Structure
---------------
State tuple = (C, n, m):
  C : [B, NH, DH_qk, DH_v]  matrix-memory (rank-1 outer product accumulator)
  n : [B, NH, DH_qk]        normalizer vector (for stable denominator)
  m : [B, NH]               scalar log-stabilizer (prevents exp overflow)

Why Matrix Memory?
------------------
Unlike scalar LSTM (sLSTM) which stores per-feature scalars, mLSTM stores
a DH_qk × DH_v matrix C per head. This allows content-based addressing:
  h_t ∝ C_t @ q_t
The query q_t acts as a "key" to retrieve relevant information from memory,
similar to attention but with recurrent accumulation.

Numerical Stability
-------------------
- Forget/input gates use log-space (softplus trick) to avoid exp(large).
- Stabilizer m_t keeps denominators well-scaled across long sequences.
- Mixed precision: compute in `compute_dtype`, store state in `state_dtype`.

Parity
------
Logic mirrors torch-native `mLSTMRecurrentKernelCell` for cross-backend testing.

[CLASS] mLSTMRecurrentKernelCell
------------------------------------------------------------
Sequential step-by-step recurrence kernel for mLSTM (no projections).

Parameters
----------
num_heads : int
    Number of attention heads (NH).
qk_dim_per_head : int
    Query/key dimension per head.
v_dim_per_head : int
    Value dimension per head.
eps : float, default 1e-6
    Numerical stability epsilon.
compute_dtype : mx.Dtype, default mx.float32
    Dtype for forward pass activations.
state_dtype : mx.Dtype, default mx.float32
    Dtype for recurrent state storage (C, n, m).

Returns (forward)
-----------------
h : mx.array [B, NH, S, DH_v]
    Hidden states for all timesteps (stacked from loop).
new_state : (C, n, m)
    Final recurrent state after processing all S steps.

[FUNCTION] __call__
------------------------------------------------------------
Execute sequential mLSTM recurrence (loop over timesteps).

Parameters
----------
q : mx.array [B, NH, S, DH_qk]
    Query projections.
k : mx.array [B, NH, S, DH_qk]
    Key projections.
v : mx.array [B, NH, S, DH_v]
    Value projections.
i_preact : mx.array [B, NH, S]
    Input gate preactivations.
f_preact : mx.array [B, NH, S]
    Forget gate preactivations.
state : tuple | None
    Previous recurrent state (C, n, m) or None for initialization.

Returns
-------
h : mx.array [B, NH, S, DH_v]
    Hidden states computed sequentially for all S timesteps.
new_state : (C, n, m)
    Final recurrent state after step S-1.



## mLSTM Forward Kernels

### The Heart of the Engine

These Metal kernels are where xLSTM-metal earns its name. After wrestling with Triton translations, numerical stability issues, and Metal's quirks, these kernels deliver the chunkwise parallel mLSTM algorithm at speeds that make the sequential fallback look silly.

**Two-Phase Chunkwise Algorithm:**

**Phase 1 - Sequential Inter-Chunk Recurrence:**
```
For chunk k = 0 to NC-1:
  C_k, n_k, m_k = recurrent_update(C_{k-1}, n_{k-1}, m_{k-1}, chunk_k)
Returns: NC+1 boundary states (initial + after each chunk)
```

**Phase 2 - Parallel Intra-Chunk Attention:**
```
For all positions i in all chunks (fully parallel):
  h_i = attention_within_chunk(i) + contribution_from_boundary(C_{k-1})
Produces: Hidden states h for all timesteps simultaneously
```

**Why This Works:**
- Inter-chunk: Sequential dependency, but only NC steps (S/L where L=chunk_size)
- Intra-chunk: No dependency, full parallelism across all chunks
- Net result: O(S·L + S/L·DH) instead of O(S²) (full attention) or O(S) sequential (too slow)

### `blocks/mlstm/mlstm_chunkwise/forward/mlstm_chunkwise_recurrent_fw_C.py`

**Recurrent state propagation across chunks.** This kernel computes the boundary states (C, n, m) sequentially, one chunk at a time. Each threadgroup processes a tile of the C matrix (which is DH_qk × DH_v per head).

**Critical numerical detail:** Uses log-space for forget/input gates to prevent exp(large) overflow. The stabilizer `m` keeps denominators well-scaled across arbitrarily long sequences.

### `blocks/mlstm/mlstm_chunkwise/forward/mlstm_chunkwise_parallel_fw_Hintra.py`

**Parallel hidden state computation.** Once we have boundary states from the recurrent kernel, this kernel computes all hidden states h in parallel. Each thread handles a position in a chunk, combining:
1. Causal attention within the chunk (lower-triangular)
2. Contribution from the prior chunk's boundary state C_{k-1}

**The speedup:** For L=64 chunks on a 2048-token sequence, this means 32 sequential steps + massive parallelism, instead of 2048 sequential steps. On Apple Silicon with unified memory, the memory bandwidth utilization is excellent.


[MODULE] mlstm_chunkwise_parallel_fw_Hintra
------------------------------------------------------------
Metal kernel for parallel part of forward pass of the mLSTM chunkwise formulation.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes outputs H within each chunk in parallel using:
1. Intra-chunk: attention within chunk using causal mask
2. Inter-chunk: contribution from previous state C_{k-1}
3. Combine: H = (H_inter + ratio * H_intra) / denom

[FUNCTION] _compile_parallel_kernel
------------------------------------------------------------
Compiler function - called once on first kernel access.

[FUNCTION] _get_kernel
------------------------------------------------------------
Get compiled kernel (compiles on first call).

[FUNCTION] mlstm_chunkwise_parallel_fw_Hintra_metal
------------------------------------------------------------
Metal kernel for parallel forward computation of mLSTM outputs within chunks.

Returns:
    matHout: (B, NH, S, DHHV)
    vecNout: (B, NH, S)
    vecMout: (B, NH, S)



### `blocks/mlstm/mlstm_chunkwise/forward/mlstm_chunkwise_recurrent_fw_C.py`


[MODULE] mlstm_chunkwise_recurrent_fw_C
------------------------------------------------------------
Metal kernel for recurrent part of forward pass of the mLSTM chunkwise formulation.
Ported from Triton to Metal C++ using mx.fast.metal_kernel().

This kernel computes inter-chunk states (C, n, m) sequentially across chunks.
Each threadgroup processes a (siz_b_DHQK, siz_b_DHHV) tile of the C matrix.

[FUNCTION] _compile_recurrent_kernel
------------------------------------------------------------
Compiler function - called once on first kernel access.

[FUNCTION] _get_kernel
------------------------------------------------------------
Get compiled kernel (compiles on first call).

[FUNCTION] mlstm_chunkwise_recurrent_fw_C_metal
------------------------------------------------------------
Metal kernel for recurrent forward computation of mLSTM chunk states.

Returns:
    matC_states: (B, NH, (NC+1)*DHQK, DHHV)
    vecN_states: (B, NH, (NC+1)*DHQK)
    scaMinter_states: (B, NH, NC+1)
    dbg_out: (L * 3 + 1,) debug buffer



---

## sLSTM

### `blocks/slstm/__init__.py`


[MODULE] __init__
------------------------------------------------------------
sLSTM MLX implementation following NCPS-style modular cells.



### `blocks/slstm/slstm_block.py`


[MODULE] slstm_block
------------------------------------------------------------
xLSTM sLSTM Cell - NCPS-compatible wrapper for sLSTM block.

This cell wraps a complete sLSTM block (sLSTM + FFN) with proper parameter
handling for model loading from safetensors and config.json.

Follows NCPS patterns for clean composability and wiring.

[CLASS] sLSTMBlock
------------------------------------------------------------
Complete xLSTM sLSTM block cell (sLSTM + FFN) for NCPS.

This represents one full xLSTM block with sLSTM:
    input -> norm_slstm -> sLSTM -> residual
          -> norm_ffn -> FFN -> residual -> output

Parameters are loaded from:
- config.json: Model hyperparameters
- safetensors: Pretrained weights (backbone.blocks.{i}.*)

Args:
    block_index: Block index for weight loading
    embedding_dim: Model dimension
    num_heads: Number of sLSTM heads
    head_dim: Dimension per head
    gate_soft_cap: Gate soft cap value
    ffn_proj_factor: FFN projection factor
    ffn_round_up_to_multiple_of: FFN dimension rounding
    norm_eps: Normalization epsilon
    use_bias: Whether to use bias
    eps: Numerical stability epsilon

[FUNCTION] __call__
------------------------------------------------------------
Forward pass through sLSTM block with residual connections.

Args:
    x: Input [B, S, D]
    state: Optional sLSTM state (c, n, m)

Returns:
    output: Output [B, S, D]
    new_state: Updated sLSTM state

[FUNCTION] get_weight_keys
------------------------------------------------------------
Get mapping of module parameters to safetensors keys.

Returns:
    Dict mapping self.parameter_path -> safetensors_key

[FUNCTION] from_config
------------------------------------------------------------
Create cell from config.json dict.

Args:
    block_index: Block index
    config: Config dict from config.json
    **kwargs: Additional arguments

Returns:
    Initialized xLSTMsLSTMCell

[FUNCTION] get_config
------------------------------------------------------------
Return configuration for serialization.



### `blocks/slstm/slstm_layers/__init__.py`


[MODULE] __init__
------------------------------------------------------------
sLSTM stepwise kernel module.



### `blocks/slstm/slstm_layers/slstm_cell.py`


[MODULE] slstm_cell
------------------------------------------------------------
Scalar LSTM (sLSTM) Cell – MLX Implementation (single-step)

Overview
--------
The scalar LSTM (sLSTM) layer is the *scalar-memory* component of xLSTM
(see Appendix A of the xLSTM paper: https://arxiv.org/pdf/2405.04517).
Instead of storing matrix-valued memory like mLSTM, each head maintains
lightweight scalar exponential statistics that enable long-range retention
with numerically stable gating.

NCPS Design Pattern
-------------------
Following NCPS / CfC style, ALL trainable parameters (projections, optional
causal conv pre-processing, group / per‑head norm, output projection) are
contained in this cell. The cell processes **one timestep** at a time:
    __call__(inputs, hx, ts) -> (output, new_hx)

Computation Flow (per timestep)
-------------------------------
1. (Optional) Causal 1D convolution over the *current* token (implemented
   via padding + conv + SiLU) for temporal mixing of i,f gate preactivations.
2. Linear projections produce:
      z_t  : candidate content         [B, NH * H]
      i_t  : input  gate preactivation [B, NH]
      f_t  : forget gate preactivation [B, NH]
      o_t  : output gate preactivation [B, NH]
3. Soft cap (cap * tanh(x / cap)) optionally applied to i_t, f_t, o_t to
   bound magnitude and stabilize exponentials.
4. Exponential stabilization:
      m_t = max(f_t + m_{t-1}, i_t)
   ensures denominators remain well‑scaled and avoids overflow/underflow.
5. Normalized gates:
      i_gate = exp(i_t              - m_t)
      f_gate = exp(f_t + m_{t-1}    - m_t)
6. State updates (per head, elementwise in H):
      c_t = f_gate * c_{t-1} + i_gate * z_t
      n_t = f_gate * n_{t-1} + i_gate
7. Normalized hidden content:
      h_tilde = c_t / (n_t + eps)
      h      = sigmoid(o_t) * h_tilde
8. Group / per‑head normalization (multi‑head layer norm) applied to h,
   flattened to [B, NH * H], then projected back to input_size.

Shapes
------
Inputs:
    inputs : [B, D]
State (hx): (c, n, m)
    c : [B, NH, H]  (content accumulator)
    n : [B, NH, H]  (normalizer accumulator)
    m : [B, NH]     (stabilizer log‑scale)
Output:
    output : [B, D]
    new_hx : (c_new, n_new, m_new)

Arguments
---------
input_size : int
    Feature dimension D.
num_heads : int
    Number of scalar heads (NH).
head_dim : int
    Per‑head hidden size (H) so NH * H = hidden_size.
conv1d_kernel_size : int, default 4
    Enables causal temporal conv for i,f gate preactivations when > 0.
use_bias : bool, default False
    Bias term in linear projections.
eps : float, default 1e-6
    Numerical stability constant for normalization.
gate_soft_cap : float, default 15.0
    Soft cap value; if None disables tanh capping.

Why Soft Capping?
-----------------
Large gate magnitudes can explode the stabilized exponentials. The soft
cap keeps preactivations in a smooth but bounded range without hard clipping.

Autograd / Numerical Notes
--------------------------
The stabilized form using m_t avoids computing exp of large positive
numbers while preserving correct ratios. Dividing by n_t + eps normalizes
cumulative weighted content, preventing scale drift.

Parity with Torch Version
-------------------------
Logic mirrors the torch_native sLSTMCell so forward parity tests can
assert closeness between MLX and PyTorch backends.

[CLASS] sLSTMCell
------------------------------------------------------------
Single‑timestep scalar LSTM (sLSTM) recurrence cell (MLX backend).

Implements one autoregressive timestep with stabilized exponential gating
and per‑head normalization. Encapsulates projections, (optional) conv
preprocessing, normalization, and output projection.

Forward Signature
-----------------
__call__(inputs, hx=None, ts=None) -> (output, new_hx)

Parameters (see module docstring for detailed semantics)
-------------------------------------------------------
input_size : int
num_heads : int
head_dim : int
conv1d_kernel_size : int, default 4
use_bias : bool, default False
eps : float, default 1e-6
gate_soft_cap : float, default 15.0

Returns
-------
output : mx.array [B, D]
    Projected hidden representation for the timestep.
new_hx : (c_new, n_new, m_new)
    Updated recurrent states.

[FUNCTION] state_size
------------------------------------------------------------
Total state size across all heads.

[FUNCTION] output_size
------------------------------------------------------------
Output size (same as input_size after out_proj).

[FUNCTION] soft_cap_gates
------------------------------------------------------------
Apply soft capping (cap * tanh(x / cap)) to gate preactivations if enabled.

[FUNCTION] __call__
------------------------------------------------------------
Run one sLSTM timestep.

Parameters
----------
inputs : mx.array [B, D]
    Current timestep features.
hx : tuple | None
    Previous state (c, n, m) or None for zero initialization.
ts : float | mx.array | None
    Optional timestep placeholder (kept for NCPS API symmetry; unused).

Returns
-------
output : mx.array [B, D]
    Hidden representation after gating, normalization, and projection.
new_hx : (c_new, n_new, m_new)
    Updated recurrent states for next step.



### `blocks/slstm/slstm_layers/slstm_neuron.py`


[MODULE] slstm_neuron
------------------------------------------------------------
sLSTM Neuron - wires together projection, kernel, and output cells.

The neuron is the complete sLSTM layer that wires together:
    Input → Projection Cell → Kernel Cell (stepwise) → Output Cell → Output

The neuron owns the wiring logic and composes the before/during/after cells.

[CLASS] sLSTMNeuron
------------------------------------------------------------
sLSTM Neuron - complete sLSTM layer with cell wiring.

Wires together the sLSTM pipeline:
1. Projection Cell: x → z, i, f, o (with optional conv)
2. Kernel Cell: z, i, f, o, state → h, new_state (stepwise recurrence)
3. Output Cell: h → output (group norm + projection)

The neuron handles sequential processing across timesteps
and composes the modular cells.

Args:
    input_size: Input dimension (embedding_dim)
    num_heads: Number of sLSTM heads
    head_dim: Hidden dimension per head
    conv1d_kernel_size: Conv kernel size (0 = disabled, default 4)
    use_bias: Whether to use bias in projections
    eps: Numerical stability epsilon
    gate_soft_cap: Soft cap value for gates (default 15.0)

[FUNCTION] state_size
------------------------------------------------------------
Return state dimensions (c, n, m).

[FUNCTION] output_size
------------------------------------------------------------
Return output dimension.

[FUNCTION] __call__
------------------------------------------------------------
Forward pass through complete sLSTM neuron.

Processes sequence step-by-step using kernel cell.

Args:
    x: Input [B, S, input_size]
    state: Optional previous state (c, n, m)
           c: [B, NH, H] - cell state
           n: [B, NH, H] - normalizer
           m: [B, NH] - stabilizer

Returns:
    output: Output [B, S, input_size]
    new_state: Updated state (c, n, m)

[FUNCTION] get_config
------------------------------------------------------------
Return configuration for serialization.



### `blocks/slstm/slstm_layers/slstm_output_cell.py`


[MODULE] slstm_output_cell
------------------------------------------------------------
sLSTM Output Cell - handles output processing.

This is the "after" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The output cell transforms kernel outputs back to input space:
- Per-head group normalization (MultiHeadLayerNorm)
- Final linear projection

It contains NO recurrence logic.

[CLASS] sLSTMOutputCell
------------------------------------------------------------
sLSTM Output Cell - output transformation only.

Handles all output processing for sLSTM:
- Per-head group normalization (MultiHeadLayerNorm)
- Final projection back to input space

No recurrence - just output transformations.

Args:
    input_size: Input dimension (for final output)
    num_heads: Number of sLSTM heads
    head_dim: Hidden dimension per head
    use_bias: Whether to use bias in output projection
    eps: Epsilon for normalization

[FUNCTION] __call__
------------------------------------------------------------
Process kernel output to final output.

Args:
    h: Hidden states from kernel [B, S, NH, H]

Returns:
    output: Final output [B, S, input_size]



### `blocks/slstm/slstm_layers/slstm_projection_cell.py`


[MODULE] slstm_projection_cell
------------------------------------------------------------
sLSTM Projection Cell - handles all input projections.

This is the "before" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The projection cell transforms raw inputs into gate pre-activations
and z (cell input candidate). It contains NO recurrence logic.

Based on canonical xLSTM sLSTMLayer structure where:
- i, f gates use conv'd input (if conv enabled)
- z, o gates use raw input
- All gates get soft-capped

[CLASS] sLSTMProjectionCell
------------------------------------------------------------
sLSTM Projection Cell - input transformation only.

Handles all input projections for sLSTM:
- Optional causal Conv1d with SiLU activation
- Gate projections: i, f (from conv'd), z, o (from raw)
- Soft capping applied to gate pre-activations

No recurrence, no output processing - just projections.

Args:
    input_size: Input dimension (embedding_dim)
    num_heads: Number of sLSTM heads
    head_dim: Hidden dimension per head
    conv1d_kernel_size: Conv kernel size (0 = disabled, default 4)
    conv_channel_mixing: Whether conv mixes channels (groups=1) or depthwise
    use_bias: Whether to use bias in z projection
    gate_soft_cap: Soft cap value for gates (default 15.0)

[FUNCTION] soft_cap
------------------------------------------------------------
Apply soft capping: cap * tanh(x / cap).

[FUNCTION] __call__
------------------------------------------------------------
Project inputs to gate pre-activations and z.

Args:
    x: Input [B, S, input_size]

Returns:
    z: Cell input candidate [B, S, NH, H]
    i_preact: Input gate pre-activation [B, S, NH] (soft-capped)
    f_preact: Forget gate pre-activation [B, S, NH] (soft-capped)
    o_preact: Output gate pre-activation [B, S, NH] (soft-capped)
    x_conv: Conv'd input (or raw if no conv) [B, S, input_size]



### `blocks/slstm/slstm_layers/stepwise/slstm_metal_kernel.py`


[MODULE] slstm_metal_kernel
------------------------------------------------------------
sLSTM Metal Kernels - Numerically Stable Implementation

Following canonical xlstm package implementation with:
- Proper stability clamps: min(exp(...), 1.0)
- logsigmoid for forget gate
- tanh(z) for cell input
- Double-double precision for critical operations

Architecture based on M2-BERT kernel patterns:
- Global kernel cache (compile once, reuse forever)
- Phase-split at barrier boundaries
- Stream chaining for parallelism
- Proper state management

[FUNCTION] _get_kernel
------------------------------------------------------------
Get or compile kernel on first use.

[FUNCTION] slstm_step_metal
------------------------------------------------------------
Single sLSTM step using Metal kernel with canonical equations.

Implements numerically stable sLSTM from xlstm package:
- logsigmoid for forget gate stability
- min(exp(...), 1.0) clamps on gates
- tanh(z) for cell input
- Division by (n + eps) for output

Args:
    z: Cell input candidate [B, NH, H]
    i_preact: Input gate pre-activation [B, NH] (soft-capped)
    f_preact: Forget gate pre-activation [B, NH] (soft-capped)
    o_preact: Output gate pre-activation [B, NH] (soft-capped)
    c_state: Cell state [B, NH, H]
    n_state: Normalizer state [B, NH, H]
    m_state: Stabilizer [B, NH]
    eps: Numerical stability epsilon

Returns:
    h_out: Hidden output [B, NH, H]
    c_state_out: Updated cell state [B, NH, H]
    n_state_out: Updated normalizer [B, NH, H]
    m_state_out: Updated stabilizer [B, NH]

[CLASS] sLSTMMetalKernel
------------------------------------------------------------
sLSTM Metal kernel wrapper for sequence processing.

Processes sequences timestep-by-timestep using canonical sLSTM equations
with proper numerical stability.

[FUNCTION] __call__
------------------------------------------------------------
Process sequence through sLSTM.

Args:
    z: Cell input [B, S, NH, H]
    i_preact: Input gate [B, S, NH]
    f_preact: Forget gate [B, S, NH]
    o_preact: Output gate [B, S, NH]
    state: Optional (c, n, m) state

Returns:
    h: Hidden states [B, S, NH, H]
    new_state: (c, n, m)



### `blocks/slstm/slstm_layers/stepwise/slstm_stepwise_kernel_cell.py`


[MODULE] slstm_stepwise_kernel_cell
------------------------------------------------------------
sLSTM Stepwise Kernel Cell - pure recurrence with Metal acceleration.

This is the "during" cell in the sLSTM pipeline:
    Input → Projection Cell → Kernel Cell → Output Cell

The stepwise kernel cell implements sequential recurrence using
Metal-accelerated kernels with canonical sLSTM equations for numerical stability.

[CLASS] sLSTMStepwiseKernelCell
------------------------------------------------------------
sLSTM Stepwise Kernel Cell - sequential recurrence only.

Implements canonical sLSTM recurrence from xlstm package with:
- Proper stability clamps: min(exp(...), 1.0)
- logsigmoid for forget gate
- tanh(z) for cell input
- Numerical stability via Metal kernels

No projections, no output processing - pure recurrence.

Args:
    num_heads: Number of sLSTM heads
    head_dim: Hidden dimension per head
    eps: Numerical stability epsilon

[FUNCTION] __call__
------------------------------------------------------------
Apply single-step sLSTM recurrence using Metal kernel.

Args:
    z: Cell input candidate [B, NH, H]
    i_preact: Input gate pre-activation [B, NH] (soft-capped)
    f_preact: Forget gate pre-activation [B, NH] (soft-capped)
    o_preact: Output gate pre-activation [B, NH] (soft-capped)
    state: Optional previous state (c, n, m)
           c: [B, NH, H] - cell state
           n: [B, NH, H] - normalizer
           m: [B, NH] - stabilizer

Returns:
    h: Hidden states [B, NH, H]
    new_state: Updated state (c, n, m)



---

## sLSTM Conv

### `blocks/slstm/slstm_layers/causal_conv1d/causal_conv1d_kernel.py`


[MODULE] causal_conv1d_kernel
------------------------------------------------------------
Causal Conv1d Cell - NCPS compliant before-cell for sLSTM.

[CLASS] CausalConv1dCell
------------------------------------------------------------
Causal Conv1d cell used in sLSTM projection (NCPS before-cell).

Args:
    channels: Feature dimension (must equal embedding dim)
    kernel_size: Temporal kernel size (0 disables conv)
    channel_mixing: If True, uses full mixing (Conv1d groups=1). Otherwise
        depthwise per-channel kernels (canonical default).



### `blocks/slstm/slstm_layers/causal_conv1d/depthwise_kernel.py`


[MODULE] depthwise_kernel
------------------------------------------------------------
Metal depthwise causal conv kernel (per-feature).

[FUNCTION] metal_causal_conv1d_depthwise
------------------------------------------------------------
Depthwise causal conv (groups=channels).



### `blocks/slstm/slstm_layers/causal_conv1d/mixing_kernel.py`


[MODULE] mixing_kernel
------------------------------------------------------------
Metal causal conv kernel with channel mixing (groups=1).

[FUNCTION] metal_causal_conv1d_mixing
------------------------------------------------------------
Causal conv with channel mixing (Conv1d groups=1).



---


---

## Implementation Notes

### What's Not Here

This document covers the production inference path. Training components (backward kernels, gradient computation) are present in the codebase but less battle-tested. The backward kernels in `mlstm_chunkwise/backward/` are ported from Triton but haven't been through the same numerical validation as the forward pass.

### Performance Characteristics

On Apple Silicon (M1 Max, M2 Ultra, etc.):
- **mLSTM chunkwise (L=64):** ~14× faster than sequential for 2048-token sequences
- **RMSNorm Metal kernel:** ~100× faster than naive MLX for 32×32 matrices
- **Mixed precision (bfloat16/float32):** Essential for memory bandwidth utilization

Unified memory architecture means data lives in one place—no host-device transfers. This is a huge advantage for recurrent models where state must persist across tokens.

### Parity Testing

Every major component has a corresponding `torch_native` implementation in the quarantine/reference code. During development, forward passes were validated against PyTorch with `rtol=1e-5` for float32, `rtol=1e-3` for bfloat16. This caught numerous subtle bugs in gate ordering, stabilizer updates, and dimension handling.

### Missing Pieces

- **Attention blocks:** Wiring detected but not implemented (sLSTM/mLSTM only)
- **Training loop:** Inference-focused; fine-tuning code exists but needs validation
- **Quantization:** No INT8/INT4 support yet (MLX has the primitives)
- **Multi-GPU:** Single-device only (MLX limitation)

### Where to Go From Here

- **Architecture overview:** See `ARCHITECTURE.md` for design philosophy
- **Numerical stability deep-dive:** See `NUMERICAL_STABILITY.md`
- **NCPS wiring details:** See `NCPS_WIRING.md`
- **Code diving:** Start at `generate.py` and follow the call stack

This is production code that runs real inference workloads. If something's documented here, it works. If it's undocumented, it might be experimental or partially implemented.

---

**Questions?** sydney@solace.ofharmony.ai
