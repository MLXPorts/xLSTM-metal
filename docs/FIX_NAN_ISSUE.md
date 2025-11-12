# NaN Issue Fix - norm_reduction_force_float32 Parameter

## Problem
When running text generation, the model crashed with:
```
ValueError: RMSNormCell produced NaNs; consider enabling norm_reduction_force_float32
```

## Root Cause
The `norm_reduction_force_float32` parameter was being read from `config.json` but was **not respecting runtime overrides** passed through the model initialization chain.

### Code Flow:
1. `generate.py` → creates `xLSTMRunner` with `norm_reduce_force_float32=True`
2. `xLSTMRunner` → creates `WiredxLSTM(norm_reduce_force_float32=True)`
3. `WiredxLSTM._build_model()` → calls `wiring.create_block_cell(norm_reduction_force_float32=True)`
4. `AutoWiring.create_block_cell()` → calls `mLSTMBlock.from_config(**kwargs)`
5. **BUG**: `mLSTMBlock.from_config()` only read from config, ignored kwargs override

## The Fix
**File**: `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_block.py`

### Fix 1: Respect runtime overrides in from_config()
**Before** (line ~364):
```python
norm_reduction_force_float32 = config.get('norm_reduction_force_float32', True)
```

**After**:
```python
norm_reduction_force_float32 = overrides.get(
    'norm_reduction_force_float32', config.get('norm_reduction_force_float32', True)
)
```

This matches the pattern used for `compute_dtype` and `state_dtype`, ensuring runtime overrides take precedence over config values.

### Fix 2: Pass parameter to mLSTMNeuron
**Before** (line ~200):
```python
self.mlstm_cell = mLSTMNeuron(
    input_size=embedding_dim,
    num_heads=num_heads,
    ...
    compute_dtype=compute_dtype,
    state_dtype=state_dtype,
)
```

**After**:
```python
self.mlstm_cell = mLSTMNeuron(
    input_size=embedding_dim,
    num_heads=num_heads,
    ...
    compute_dtype=compute_dtype,
    state_dtype=state_dtype,
    force_float32_reductions=norm_reduction_force_float32,
)
```

The `mLSTMNeuron` contains an `mLSTMOutputCell` which uses `MultiHeadRMSNormCell`. Without this parameter, the output cell's norm was NOT using float32 reductions, causing NaNs.

## Why This Matters
When using **bfloat16** for compute (`autocast_kernel_dtype: "bfloat16"`), RMSNorm reductions can accumulate precision errors over long feature dimensions (e.g., 4096). 

Setting `norm_reduction_force_float32=True` ensures:
- Squared values are accumulated in **float32** precision
- Final RMS computation avoids drift/overflow
- Output is cast back to compute dtype (bfloat16)

### Where NaNs Occurred
The error appeared in the **mLSTM output cell's MultiHeadRMSNorm** (not the block-level pre-norms). This happened because:

1. The `norm_mlstm` and `norm_ffn` (block-level) were correctly configured with `force_float32_reductions=True`
2. BUT the `mLSTMNeuron.output_cell.norm` (inside the neuron) was NOT receiving this parameter
3. After several blocks of bfloat16 computation, the accumulated error in the multi-head norm produced NaNs

Both fixes were needed:
- **Fix 1**: Ensures overrides work (for future configurability)
- **Fix 2**: Ensures the parameter reaches ALL RMSNorm layers, including nested ones

## Testing
Run generation to verify no NaNs:
```bash
python generate.py --model xlstm_7b_model --prompt "Hello" --max-tokens 10
```

Should now complete without ValueError.

## Related Config
From `xlstm_7b_model/config.json`:
```json
{
  "autocast_kernel_dtype": "bfloat16",      // Compute in bf16
  "inference_state_dtype": "float32",        // State in fp32
  "norm_reduction_force_float32": true       // Force fp32 in norm reductions
}
```

The fix ensures all three settings are properly respected throughout the initialization chain.

