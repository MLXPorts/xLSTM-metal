# Fix Summary: Dtype Configuration Issue

## Problem
Model crashed with `ValueError: RMSNormCell produced NaNs` when generating text with real tokenized input (but worked with dummy input).

## Root Cause
The xLSTMRunner was incorrectly using `autocast_kernel_dtype: "bfloat16"` as the compute dtype for **all model weights and activations**, when it should have been using `torch_dtype: "float32"`.

### Config Structure
The xLSTM config.json has two separate dtype fields with different purposes:

1. **`torch_dtype: "float32"`** - Base dtype for model weights and general computation
2. **`autocast_kernel_dtype: "bfloat16"`** - Dtype for specific optimized kernel operations only

## The Bug
In `xlstm_metal/mlx_jit/generate.py` line 64:

**Before (incorrect):**
```python
self.compute_dtype = resolve_dtype(self.config.get('autocast_kernel_dtype'))
```

This caused all weights (embeddings, Linear layers, etc.) to be loaded/cast to bfloat16, which created numerical instability leading to NaN values in the mLSTM layer outputs.

## The Fix
**After (correct):**
```python
# Use torch_dtype for model weights (float32), not autocast_kernel_dtype (bfloat16)
# autocast_kernel_dtype is for specific kernel operations only
self.compute_dtype = resolve_dtype(self.config.get('torch_dtype', 'float32'))
```

Now the model uses float32 for weights and activations, which matches the canonical transformers implementation.

## Why It Matters
- **bfloat16** has limited precision (7-bit mantissa) which can cause accumulated errors in deep networks
- **float32** provides stable computation for the 32-layer xLSTM-7B model
- The `autocast_kernel_dtype` setting is meant for *selective* mixed-precision in specific kernels, not global dtype

## Additional Fixes Made
During debugging, we also fixed:

1. **`mLSTMBlock.from_config()`** - Now properly respects `norm_reduction_force_float32` override from kwargs
2. **`mLSTMNeuron` initialization** - Now receives `force_float32_reductions` parameter to pass to output cell's MultiHeadRMSNorm

## Testing
```bash
# Short generation test
python generate.py --model xlstm_7b_model --prompt "Hello world" --max-tokens 20

# Original failing prompt
python generate.py --model xlstm_7b_model --prompt "User: Who are you?\nAssistant: " --max-tokens 50
```

**Result:** ✅ Successful generation without NaN errors!

Both tests complete successfully. The model now runs stably with float32 weights.

## Files Modified
1. `xlstm_metal/mlx_jit/generate.py` - Fixed dtype selection
2. `xlstm_metal/mlx_jit/blocks/mlstm/mlstm_block.py` - Fixed parameter propagation
3. Multiple debug enhancements (can be removed or kept for diagnostics)

## Lesson Learned
Always distinguish between:
- **Model dtype** (`torch_dtype`) - for weights/activations
- **Kernel dtype** (`autocast_kernel_dtype`) - for specific optimized ops
- **State dtype** (`inference_state_dtype`) - for recurrent state

Using the wrong one can cause subtle numerical issues that only appear with real data!

