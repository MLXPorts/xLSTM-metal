# Inference and Memory Status

## Inference construction

This Apple port uses sLSTM for training work and mLSTM for inference. Framework choice for training is separate from those cell roles. TiRex and other xLSTM-derived architectures have their own recipes.

The current [WiredxLSTM builder](../xlstm_metal/mlx_jit/models/wired_xlstm.py) constructs mLSTM blocks. It raises `NotImplementedError` for sLSTM and attention blocks. [AutoWiring](../xlstm_metal/mlx_jit/wiring/auto_wiring.py) can detect those block types, but detection is separate from model construction. The presence of sLSTM modules or backward kernels does not make them part of the inference stack.

Blocks run sequentially: each block consumes the previous block's output. Inside mLSTM, [mLSTMNeuron](../xlstm_metal/mlx_jit/blocks/mlstm/mlstm_chunkwise/mlstm_neuron.py) uses Metal chunkwise kernels for full chunks and the sequential recurrent kernel for the remaining tokens. Parallel work within chunks, heads, and tiles does not remove the recurrence between chunk boundaries.

## Recurrent state

Each block returns `(C, n, m)`:

- `C`: `[batch, heads, qk_dim_per_head, v_dim_per_head]`, the gated key/value outer-product accumulator.
- `n`: `[batch, heads, qk_dim_per_head]`, the normalizer vector.
- `m`: `[batch, heads]`, the log stabilizer.

The [recurrent cell](../xlstm_metal/mlx_jit/blocks/mlstm/mlstm_chunkwise/mlstm_recurrent_kernel_cell.py) reads the matrix with the current query and returns all three tensors. The model accepts and returns a list of these tuples, one per block. Compute and state dtypes can be configured independently.

The [runner](../xlstm_metal/mlx_jit/generate.py) retains state through `generate_next_token`. Each `generate` call resets state before processing its prompt. Cross-turn continuity therefore requires explicit state management rather than repeated `generate` calls.

This is fixed-size associative recurrent memory. It accumulates and forgets information; it does not provide an episode index, an external retrieval store, or an inference snapshot/save-restore API. Experiments that capture state must retain `C`, `n`, and `m` for every block.

## Planned memory extensions

The [extended-context plan](../docs_archive/research/EXTENDED_CONTEXT_PLAN.md) proposes hierarchical summaries, recent/relevant/compressed/persistent pools, salience budgets, and recall evaluation. These are research directions, not options implemented in the current inference runner. The plan's sLSTM scheduling ideas are distinct from the deployed mLSTM inference path.

The [state-expansion notes](../docs_archive/research/STATE_EXPANSION_PRECISION.md) describe a proposed precision representation. The current recurrent cells store ordinary arrays in `state_dtype`.
