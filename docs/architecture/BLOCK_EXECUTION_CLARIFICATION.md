# xLSTM Block Execution: Sequential vs Parallel

**Date:** 2025-01-29  
**Clarification:** Understanding "parallel" in xLSTM context

## Misunderstanding Cleared

Initial confusion: "two mLSTM blocks" - this was incorrect interpretation.

## Actual Architecture (from transformers modeling_xlstm.py)

### Block Structure

```python
class xLSTMBlock(nn.Module):
    def __init__(self, config):
        self.norm_mlstm = RMSNorm(...)
        self.mlstm_layer = xLSTMLayer(config)  # ← ONE mLSTM per block
        self.norm_ffn = RMSNorm(...)
        self.ffn = xLSTMFeedForward(config)
    
    def forward(self, x, state):
        # mLSTM path with residual
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm
        
        # FFN path with residual
        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn
        
        return x, state
```

### Model Execution (from modeling_xlstm.py lines 1459-1469)

```python
# 32 blocks executed SEQUENTIALLY (not parallel)
for layer_idx, xlstm_block in enumerate(self.blocks):
    hidden_states, rnn_state = xlstm_block(
        hidden_states,
        state=cache_params.rnn_state[layer_idx]
    )
```

**KEY:** Blocks execute **sequentially** because each block's output is the next block's input.

## What "Parallel" Actually Means

### 1. Chunkwise Parallel (Within mLSTM)

From `mlstm_chunkwise_parallel_fw_H` (modeling_xlstm.py line 157):

```python
def mlstm_chunkwise_parallel_fw_H(
    matQ, matK, matV, matC_states, vecN_states, scaMinter_states,
    vecI, vecB, qk_scale, chunk_size=64, num_chunks=1, eps=1e-6
):
    """
    Compute outputs WITHIN each chunk in parallel.
    
    Steps:
    1. Split sequence into chunks: S = num_chunks * chunk_size
    2. Compute inter-chunk states (sequential, one per chunk)
    3. Compute intra-chunk outputs (PARALLEL across chunks)
    
    Parallelism: GPU processes all tokens within a chunk simultaneously
    Sequential: Must process chunks in order (state dependency)
    """
    # Reshape into chunks
    matQ = matQ.view(batch_size, nh, nc, chunk_size, dqk)
    matK = matK.view(batch_size, nh, nc, chunk_size, dqk)
    
    # Intra-chunk attention (parallel within chunk)
    matS_chunk = (matQ @ matK.transpose(-2, -1)) * qk_scale
    
    # Combine with inter-chunk state
    matH_out = ...
```

**Parallelism = GPU tiling/vectorization**, NOT multiple mLSTM blocks.

### 2. GPU-Level Parallelism (Metal Kernels)

Our Metal kernels already implement this:

```
fw_kernel_parallel.metal:
- Threadgroup: (siz_b_DHHV, siz_b_LQ, 1)
- Grid: (num_tiles_DHHV, num_tiles_LQ, NC * B * NH)
- Processes multiple chunks/heads/batches in parallel on GPU
```

**This is tile-based parallel computation** - standard GPU optimization.

## Three Types of "Parallel" in xLSTM

### 1. **Block-Level: SEQUENTIAL** ❌

```
Block 0 → Block 1 → Block 2 → ... → Block 31
```

Sequential because of state dependencies (each block needs previous block's output).

### 2. **Chunk-Level: HYBRID** ✅

```
Chunk 0 (recurrent) → Chunk 1 (recurrent) → ... → Chunk N (recurrent)
  ↓                     ↓                           ↓
[tokens 0-63]      [tokens 64-127]           [tokens S-64:S]
  parallel            parallel                   parallel
```

- **Between chunks:** Sequential (state dependency)
- **Within chunk:** Parallel (GPU processes all tokens together)

### 3. **GPU-Level: PARALLEL** ✅

```
Metal Kernel Execution:
┌───────────────────────────────────────────┐
│  Threadgroup 0    Threadgroup 1    ...    │
│  ┌──────────┐    ┌──────────┐            │
│  │ Thread 0 │    │ Thread 0 │            │
│  │ Thread 1 │    │ Thread 1 │            │
│  │   ...    │    │   ...    │            │
│  └──────────┘    └──────────┘            │
└───────────────────────────────────────────┘
```

All threads execute in parallel on GPU cores.

## Canonical Implementation Pattern

```python
# From mlstm_backend_integration.py (transformers)
def wrap_chunkwise_pad_zeros(...):
    """
    Chunkwise kernel processes chunks in parallel.
    
    Decision logic:
    - S % chunk_size == 0: Use chunkwise parallel
    - S % chunk_size != 0: Pad to multiple, then chunkwise
    - S == 1: Single-step recurrent
    """
```

### Kernel Selection (from config.json)

```json
{
  "chunkwise_kernel": "chunkwise--triton_xl_chunk",  // ← Parallel within chunks
  "sequence_kernel": "native_sequence__triton",      // ← Sequential fallback
  "step_kernel": "triton",                           // ← Single-step recurrent
  "chunk_size": 64
}
```

**Translation:**

- `chunkwise_kernel` = Our `mlstm_chunkwise` (uses `fw_kernel_parallel.metal`)
- `sequence_kernel` = Our `mlstm_sequential` (loop with recurrent steps)
- `step_kernel` = Our `mlstm_recurrent_step` (uses `fw_kernel_recurrent.metal`)

## Our Implementation Status

### ✅ Already Implemented

1. **Sequential blocks** - `xLSTMBlock` in `xlstm_metal/blocks/mlx/mlstm/xlstm_block.py`
2. **Chunkwise parallel** - `mlstm_chunkwise` in `kernel.py` using Metal parallel kernels
3. **GPU parallelism** - Metal kernels with threadgroups/tiling

### ❌ Missing (Not Actually Needed)

- **Parallel block execution** - Not possible due to state dependencies
- **Multiple mLSTM per block** - Canonical doesn't do this

## Conclusion

**There are NO multiple mLSTM blocks running in parallel.**

The "parallel" you saw refers to:

1. **Intra-chunk parallelism** - GPU processes all tokens in a chunk simultaneously
2. **Metal kernel tiling** - GPU threadgroups process tiles in parallel

Both are **already implemented** in our Metal kernels (`fw_kernel_parallel.metal`).

The 32 blocks execute **sequentially** because:

- Block N+1 needs Block N's output as input
- RNN state flows from block to block
- This is fundamental to the architecture

## What We Actually Need

1. ✅ **Sequential block iteration** - Already have this
2. ✅ **Chunkwise parallel kernels** - Already have Metal implementation
3. ⏭️ **Dtype management fix** - HIGH PRIORITY
4. ⏭️ **Strategy registry** - NCPS pattern for kernel selection
5. ⏭️ **Component extraction** - Gates, projections, norms as reusable components

**No changes needed for "parallel" - we already have it via GPU tiling!**
