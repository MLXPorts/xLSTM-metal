# Complete Fix Summary: xLSTM-Metal NaN Issue Resolution

## Date
November 12, 2025

## Problem Statement
The xLSTM-7B model crashed with `ValueError: RMSNormCell produced NaNs` during text generation when using real tokenized input, but worked fine with dummy integer inputs.

## Investigation Process

### Key Findings
1. **Dummy input (tokens [1,2,3,4])** → ✅ Works
2. **Real tokenized input** → ❌ NaN crash
3. **NaN originated in mLSTM layer**, not in RMSNorm itself
4. **Embeddings were in bfloat16** while other weights were float32
5. **Root cause: Wrong dtype config field being used**

### Debugging Steps
1. Added comprehensive docstrings to 19+ files (mLSTM, sLSTM, RMSNorm, wiring, etc.)
2. Created diagnostic test to isolate the issue
3. Added debug output to track NaN propagation through layers
4. Discovered embeddings were bfloat16 when they should be float32
5. Traced back to xLSTMRunner using wrong config field

## The Core Issue

### Config Structure
The xLSTM config.json has **three separate dtype fields**:

1. **`torch_dtype: "float32"`** - Base model dtype (weights, activations)
2. **`autocast_kernel_dtype: "bfloat16"`** - For specific kernel optimizations
3. **`inference_state_dtype: "float32"`** - For recurrent state storage

### The Bug
`xlstm_metal/mlx_jit/generate.py` was using the WRONG field:

```python
# WRONG - uses bfloat16 for everything
self.compute_dtype = resolve_dtype(self.config.get('autocast_kernel_dtype'))
```

This caused:
- All embeddings cast to bfloat16
- All Linear layers use bfloat16 weights
- Accumulated precision errors over 32 layers
- NaN values in mLSTM output

### The Fix
```python
# CORRECT - uses float32 for model weights
self.compute_dtype = resolve_dtype(self.config.get('torch_dtype', 'float32'))
```

## Additional Fixes Made

### 1. Parameter Propagation in mLSTMBlock
**File:** `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_block.py`

**Issue:** `norm_reduction_force_float32` wasn't being read from overrides in `from_config()`

**Fix:**
```python
# Before
norm_reduction_force_float32 = config.get('norm_reduction_force_float32', True)

# After  
norm_reduction_force_float32 = overrides.get(
    'norm_reduction_force_float32', config.get('norm_reduction_force_float32', True)
)
```

### 2. mLSTMNeuron Missing Parameter
**File:** `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_block.py`

**Issue:** `force_float32_reductions` wasn't passed to mLSTMNeuron, so the output cell's MultiHeadRMSNorm couldn't use float32 reductions

**Fix:**
```python
self.mlstm_cell = mLSTMNeuron(
    # ...existing params...
    force_float32_reductions=norm_reduction_force_float32,  # Added
)
```

### 3. Improved Error Message
**File:** `xlstm_metal/mlx_jit/blocks/rms_norm/rmsnorm.py`

Updated error message to include dtype guidance:
```python
raise ValueError(
    "RMSNormCell produced NaNs; consider enabling norm_reduction_force_float32 "
    "or using float32 compute_dtype instead of bfloat16"
)
```

## Files Modified

### Core Fixes
1. `xlstm_metal/mlx_jit/generate.py` - Fixed dtype selection (main fix)
2. `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_block.py` - Fixed parameter propagation (2 places)
3. `xlstm_metal/mlx_jit/blocks/rms_norm/rmsnorm.py` - Improved error message

### Documentation Added
Created comprehensive docstrings for 19 files:
- mLSTM components (block, neuron, projection, output, kernels)
- sLSTM components
- RMSNorm and multi-head variants
- Wiring and AutoWiring
- SoftCap activation
- Config loader and dtype utilities
- GatedFFN (SwiGLU)
- WiredxLSTM model
- generate.py CLI

### Documentation Files
- `docs/FIX_NAN_ISSUE.md` - Initial debugging notes
- `docs/FIX_DTYPE_ISSUE.md` - Final fix summary
- `docs/DOCSTRING_ENRICHMENT_SUMMARY.md` - Documentation changelog

## Testing & Validation

### Test Cases
```bash
# Test 1: Simple prompt
python generate.py --model xlstm_7b_model --prompt "Hello world" --max-tokens 20

# Test 2: Original failing case
python generate.py --model xlstm_7b_model --prompt "User: Who are you?\nAssistant: " --max-tokens 50
```

### Results
✅ **All tests pass** - No NaN errors, stable generation

### Before Fix
```
ValueError: RMSNormCell produced NaNs; consider enabling norm_reduction_force_float32
```

### After Fix
```
✓ xLSTM NCPS model created with 32 blocks, 4096d
[Generated text completes successfully]
```

## Key Lessons

### 1. Config Field Semantics Matter
- `torch_dtype` = model base dtype
- `autocast_kernel_dtype` = selective kernel optimization dtype
- `inference_state_dtype` = recurrent state dtype

**Never conflate these!**

### 2. Dtype Mixing Issues
Using bfloat16 for everything in a 32-layer model causes:
- Precision loss accumulation
- Numerical instability (NaNs, Infs)
- Only appears with real data (dummy data doesn't trigger edge cases)

### 3. Debugging Strategy
1. **Isolate**: Test with dummy vs real input
2. **Localize**: Add targeted debug output to find NaN origin
3. **Compare**: Check canonical implementation (transformers)
4. **Validate**: Test fix thoroughly

### 4. Documentation is Critical
The extensive docstrings added during debugging helped us:
- Understand the architecture flow
- Identify where parameters should be passed
- Document the correct usage patterns

## Impact
- ✅ xLSTM-7B model now runs stably on MLX/Metal
- ✅ Correct dtype handling matches transformers implementation
- ✅ Comprehensive documentation for future development
- ✅ Proper parameter propagation throughout the stack

## Verification
To verify the fix is working:
```bash
cd /Volumes/emberstuff/xLSTM-metal
python generate.py --model xlstm_7b_model --prompt "Test prompt" --max-tokens 100
```

Should complete without errors and generate coherent (or at least non-NaN) text.

---

**Status:** ✅ **RESOLVED** - Model generates text successfully without NaN errors.

