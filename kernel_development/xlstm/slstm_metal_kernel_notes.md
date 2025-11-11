# sLSTM Metal Kernel Implementation Notes

## Overview

Successfully implemented a numerically stable sLSTM Metal kernel that matches the canonical reference implementation from the xlstm package.

## Architecture

### Modular Cell Design (NCPS Pattern)
```
Input → Projection Cell → Kernel Cell → Output Cell → Output
```

1. **Projection Cell** (`slstm_projection_cell.py`)
   - Optional Conv1d with SiLU activation
   - Gate projections: i, f (from conv'd input), z, o (from raw input)
   - Soft capping applied to gate pre-activations

2. **Kernel Cell** (`slstm_stepwise_kernel.py`)
   - Pure recurrence using Metal-accelerated kernels
   - Single timestep processing
   - Canonical sLSTM equations with numerical stability

3. **Output Cell** (`slstm_output_cell.py`)
   - MultiHeadLayerNorm (group normalization)
   - Output projection back to input space

4. **Neuron** (`slstm_neuron.py`)
   - Wires all cells together
   - Handles sequential timestep processing

## Metal Kernel Implementation

### Canonical Equations (from xlstm package)

```python
# Line 24: logfplusm = m + logsigmoid(fraw)
# Line 28: mnew = max(iraw, logfplusm)
# Line 30: igate = min(exp(iraw - mnew), 1.0)
# Line 31: fgate = min(exp(logfplusm - mnew), 1.0)
# Line 32: cnew = fgate * c + igate * tanh(zraw)
# Line 33: nnew = fgate * n + igate
# Line 34: ynew = ogate * cnew / nnew
```

### Numerical Stability Features

1. **logsigmoid for forget gate**
   ```metal
   inline float logsigmoid(float x) {
       if (x >= 0.0f) return -log(1.0f + exp(-x));
       else return x - log(1.0f + exp(x));
   }
   ```

2. **Exponential clamps**
   ```metal
   float i_gate = min(exp(i_raw - m_new), 1.0f);
   float f_gate = min(exp(logfplusm - m_new), 1.0f);
   ```

3. **Stable tanh**
   ```metal
   inline float stable_tanh(float x) {
       if (x > 10.0f) return 1.0f;
       if (x < -10.0f) return -1.0f;
       return tanh(x);
   }
   ```

4. **Division with epsilon**
   ```metal
   float h_val = o_gate * c_new / (n_new + eps);
   ```

### Double-Double Precision Helpers

For critical operations requiring extended precision:
```metal
struct dd_t { float hi; float lo; };
inline dd_t two_sum(float a, float b) { ... }
inline dd_t two_prod(float a, float b) { ... }
inline dd_t dd_add(dd_t a, dd_t b) { ... }
```

## Critical Bug Fix: Grid/Threadgroup Configuration

### The Problem

Initial implementation used:
```python
num_groups = mx.divide(mx.add(u32(total_heads), mx.subtract(tpg, one)), tpg)
grid = (num_groups, one, one)
threadgroup = (tpg, one, one)
```

This resulted in:
- `num_groups` as float32 (1.01172) instead of uint32
- Only thread 0 executing, all other threads idle
- Multi-head processing completely broken

### The Solution

MLX Metal kernels require `grid` and `threadgroup` to **match** for single-threadgroup execution:

```python
# Fixed configuration
grid = (256, 1, 1)
threadgroup = (256, 1, 1)
```

This launches 256 threads in one threadgroup. Threads 0 to (B*NH-1) do work, the rest return early.

### Key Insight

Unlike standard Metal dispatch where:
- `grid` = number of threadgroups
- `threadgroup` = threads per threadgroup

MLX `mx.fast.metal_kernel` requires:
- `grid` = `threadgroup` for single-threadgroup kernels
- Both parameters specify the threadgroup size

## Testing Results

All tests pass with float32 precision tolerance (< 1e-5):

```
Testing single sLSTM timestep...
  Max h difference: 1.19e-07 ✓
  Max c difference: 1.19e-07 ✓
  Max n difference: 2.38e-07 ✓
  Max m difference: 0.00e+00 ✓

Testing zero initial state...
  Max h difference: 5.96e-08 ✓

Testing numerical stability...
  Max h difference: 5.96e-08 ✓
  No NaN/Inf detected ✓

Testing sequential recurrence...
  Max difference across 5 steps: 8.94e-08 ✓
```

## M2-BERT Kernel Patterns Applied

1. **Global kernel cache** - compile once, reuse forever
   ```python
   _KERNELS = {}
   def _get_kernel(name):
       if name not in _KERNELS:
           _KERNELS[name] = mx.fast.metal_kernel(...)
       return _KERNELS[name]
   ```

2. **Double-double precision** for critical accumulations

3. **Proper threadgroup barriers** at synchronization points

4. **Stream chaining** for parallelism (to be implemented for sequence-level parallelism)

## Files Created/Modified

### Created
- `slstm_metal_kernel.py` - Metal kernel implementation
- `slstm_stepwise_kernel.py` - Kernel cell wrapper
- `typed.py` - Type helpers for Metal dispatch
- `slstm_projection_cell.py` - Before cell
- `slstm_output_cell.py` - After cell
- `slstm_neuron.py` - Neuron wiring
- `test_slstm_metal_kernel.py` - Comprehensive test suite
- `kernel_development/m2bert_metal_kernels/` - M2-BERT reference kernels

### Modified
- Various `__init__.py` files for module exports

## Next Steps (Future Work)

1. **Integrate Metal CausalConv1dLayer** - replace `nn.Conv1d` in projection cell
2. **Stream chaining for sequences** - parallelize across timesteps using M2-BERT patterns
3. **Multi-threadgroup support** - for very large batch/head combinations
4. **Benchmark vs pure MLX** - measure Metal kernel speedup
5. **Test on real models** - validate with pre-trained xLSTM models

## References

- Canonical sLSTM: `/Users/sydneybach/miniconda3/lib/python3.12/site-packages/xlstm/blocks/slstm/src/vanilla/slstm.py`
- M2-BERT kernels: `/Volumes/stuff/Projects/m2-bert-mlx/bert/src/mm_mlx/`
- xlstm package: https://github.com/NX-AI/xlstm

---

**Status**: ✓ Complete and tested
**Date**: 2025-11-10
