# Numerical Stability in xLSTM-Metal

**Author:** Sydney Renee  
**Date:** January 2025  
**Status:** Production lessons learned

This document captures every numerical pitfall encountered during the port. Read this before touching any math operations.

## Executive Summary

**You will have NaN problems. Here's why and how to fix them.**

This port required solving:
1. Dtype confusion (3 config fields, used wrong one)
2. FFT normalization differences (PyTorch vs MLX)
3. Python scalar contamination (float()/int() everywhere)
4. State shape errors (k⊗v vs v⊗k silently wrong)
5. Forget gate formula (logsigmoid vs log+sigmoid)
6. Denominator stabilization (when to add eps)

All discovered the hard way. Learn from my mistakes.

## Problem 1: The Three-Dtype Config

### Symptom

Model works with dummy input `[1,2,3,4]` but crashes with real tokenized text:

```
ValueError: RMSNormCell produced NaNs
```

### Root Cause

xLSTM config.json has THREE dtype fields:

```json
{
  "torch_dtype": "float32",              // Model weights
  "autocast_kernel_dtype": "bfloat16",   // Selective kernel ops
  "inference_state_dtype": "float32"     // Recurrent state
}
```

**The Bug:**
```python
# WRONG - uses bfloat16 for EVERYTHING
self.compute_dtype = resolve_dtype(config['autocast_kernel_dtype'])
```

This cast all weights to bf16. After 32 layers with 4096 dimensions, accumulated precision loss → NaNs.

**The Fix:**
```python
# CORRECT - uses float32 for model
self.compute_dtype = resolve_dtype(config.get('torch_dtype', 'float32'))
```

### Why BFloat16 Failed

BFloat16 has:
- 8-bit exponent (same range as float32)
- **7-bit mantissa** (vs 23-bit in float32)

For a 32-layer model with 4096-dimensional vectors:
- Each layer accumulates ~4096 additions
- Relative error per op: ~1e-2 (bf16) vs ~1e-7 (fp32)
- After 32 layers: errors compound to order-one magnitude

**Result:** Hidden states drift, then explode to NaN in later blocks.

### Lesson

**Know which dtype field controls what:**
- `torch_dtype` → Model weights and activations (use this!)
- `autocast_kernel_dtype` → Specific fused ops (ignore for now)
- `inference_state_dtype` → Recurrent (C, n, m) storage (keep fp32)

**Never assume dtype semantics.** Check canonical implementation.

**References:**
- `docs_archive/COMPLETE_FIX_SUMMARY.md`
- `docs_archive/FIX_DTYPE_ISSUE.md`

## Problem 2: FFT Normalization (MLX vs PyTorch)

### Symptom

Hyena/Monarch convolution outputs off by factor of 2-4. Parity tests fail.

### Root Cause

PyTorch and MLX handle `irfft` normalization differently:

**PyTorch:**
```python
# Manual scaling on forward transform
k_f = torch.fft.rfft(k, n=2*L) / (2*L)
y = torch.fft.irfft(u_f * k_f, n=2*L, norm='forward')
# norm='forward' means NO scaling on inverse
```

**MLX:**
```python
# No manual scaling needed
k_f = mx.fft.rfft(k, n=2*L)
y = mx.fft.irfft(u_f * k_f, n=2*L)
# MLX applies 1/n automatically on inverse
```

**If you mix these patterns:**
- Missing `1/n` → outputs too large
- Double `1/n` → outputs too small
- Either way: order-of-magnitude errors

### The Rule

**Exactly ONE normalization factor must appear:**
- PyTorch: Put it on forward (`/n`) OR inverse (`norm='ortho'`)
- MLX: Put it ONLY on forward, inverse does it automatically
- **Never on both**

### How We Found This

Used tracer in `experiments/lab/trace_monarch_divergence.py`:
```
Stage         max_abs    rel_err   
──────────────────────────────────
...
fft_output    5.32e-06   4.82e-07   ← After fix
fft_output    2.14e-01   8.91e-01   ← Before (wrong!)
```

Factor of ~40,000 error from missing/doubled normalization.

### Lesson

**Test FFT paths explicitly:**
1. Generate random input
2. Transform → multiply → inverse
3. Check parseval (energy conservation)
4. Compare with NumPy reference (but watch for fp64 promotion!)

**Never trust framework defaults match.** Always verify.

**Reference:** `docs_archive/NUMERIC_STABILITY_TORCH_vs_MLX.md`

## Problem 3: Python Scalars Break Everything

### Symptom

- Profiler shows high CPU usage in GPU kernels
- Numerical drift worse than expected
- Graph compilation warnings

### Root Cause

Code full of:
```python
scale = float(some_tensor)          # Sync to host, convert to Python float
result = input * scale              # Re-upload, but now double-rounded
```

**What `float(tensor)` does:**
1. Synchronize GPU → CPU (blocks)
2. Convert to Python float (host fp64)
3. Round to fp64
4. Re-upload to GPU as fp32
5. **Second rounding:** fp64 → fp32

**Why this matters:**
- Graph breaks (lazy evaluation lost)
- Double rounding changes results
- Metal buffer allocs/deallocs
- At 300M ops/sec over 24hr: compounds to measurable drift

### The Fix

**Ban these everywhere:**
```python
# FORBIDDEN
x = float(tensor)
x = int(tensor)
x = tensor.item()
x = tensor.tolist()[0]
```

**Use backend scalars:**
```python
# CORRECT
scale = mx.array(0.5, dtype=mx.float32)
result = mx.multiply(input, scale)
```

**Use backend ops:**
```python
# FORBIDDEN
result = a + b * c / d

# CORRECT
result = mx.divide(mx.add(a, mx.multiply(b, c)), d)
```

### Enforcement

Run `emberlint.py` before every commit:
```bash
python tools/emberlint.py --verbose xlstm_metal/
```

Catches:
- Python int() / float() casts
- Arithmetic operators (+, -, *, /, %, **)
- .item() calls
- NumPy contamination

**Zero tolerance.** No exceptions except loop indices.

### Exception

**Loop variables are fine:**
```python
for i in range(num_blocks):  # i is Python int - OK
    process_block(i)
```

This is configuration, not computation.

**Reference:** `docs_archive/NUMERIC_STABILITY_TORCH_vs_MLX.md`

## Problem 4: State Shape Matters

### Symptom

Model runs, no errors, but generates garbage. Debug output shows activations in reasonable range.

### Root Cause

Covariance update formula has specific tensor product order:

**CORRECT:**
```python
C_new = f * C_old + i * (k[:,:,:,None] @ v[:,:,None,:])
# Shape: [B, NH, QK_DH, V_DH]
# k⊗v: key outer value
```

**WRONG:**
```python
C_new = f * C_old + i * (v[:,:,:,None] @ k[:,:,None,:])
# Shape: [B, NH, V_DH, QK_DH]  ← BACKWARDS
# v⊗k: value outer key
```

**Why this is silent:**
- Shape is still 4D tensor
- No runtime error
- Math executes fine
- Output is just completely wrong

Later retrieval does `Q @ C @ V` which is now nonsense, but doesn't crash.

### How to Catch

**Shape assertions:**
```python
assert C.shape == (B, NH, qk_dim, v_dim), f"C shape wrong: {C.shape}"
```

**Canonical reference:**
Check transformers implementation line-by-line:
```python
# From modeling_xlstm.py:421
matC_state_new = ... (key[:, :, :, None] @ value[:, :, None, :])
# key: [B, NH, QK_DH]
# value: [B, NH, V_DH]
# Result: [B, NH, QK_DH, V_DH]
```

**Parity tests:**
Compare every intermediate tensor, not just final output.

### Lesson

**Tensor dimensions are NOT documentation.**

Matrix products are **not commutative**. `A @ B ≠ B @ A` even if shapes "work".

**Always cross-reference canonical implementation** for exact einsum/matmul patterns.

**Reference:** `docs_archive/architecture/MLSTM_NUMERICAL_STABILITY_ANALYSIS.md`

## Problem 5: Forget Gate Formula

### Symptom

Numerical instability in long sequences. State values explode or vanish.

### Root Cause

Forget gate needs `logsigmoid`, not `log(sigmoid(...))`:

**UNSTABLE:**
```python
f = mx.sigmoid(f_preact)     # Can underflow to 0
f_log = mx.log(f)            # log(0) = -inf
```

**STABLE:**
```python
# logsigmoid(x) = -log(1 + exp(-x))
f_log = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_preact)))))
```

**Why:**
- For large negative x: `sigmoid(x) → 0`, then `log(0) → -inf`
- `logsigmoid(x)` stays finite: `≈ x` for large negative x

### The Pattern

**Max-stabilized exponentials:**
```python
# Update max state FIRST
m_new = mx.maximum(mx.add(f_log, m_old), i_gate)

# Then exponentiate RELATIVE to max
f_exp = mx.exp(mx.subtract(mx.add(f_log, m_old), m_new))
i_exp = mx.exp(mx.subtract(i_gate, m_new))
```

This ensures `exp(x - m_new) ≤ 1.0` always.

**Why it matters:**
- Without max: `exp(large_value) → inf`
- With max: `exp(x - max(x)) ≤ 1.0` by definition

### Lesson

**Exponentials need stabilization.**

Never `exp(x)` directly in recurrent loops. Always:
1. Track running max
2. Exponentiate relative to max
3. Use logsigmoid, not log+sigmoid

**Reference:** `docs_archive/architecture/MLSTM_NUMERICAL_STABILITY_ANALYSIS.md` lines 14-50

## Problem 6: When to Add Epsilon

### Symptom

Division by zero errors or NaNs in retrieval.

### Root Cause

Denominator can go to zero in edge cases:
```python
h_num = Q @ C        # Numerator
h_den = Q @ n        # Denominator (can be ~0)
h = h_num / h_den    # NaN!
```

**WRONG:**
```python
h_den = Q @ n + eps  # Add eps always
```

This adds eps even when denominator is large, biasing results.

**CORRECT:**
```python
max_val = mx.exp(mx.negative(m))  # Lower bound
h_den = mx.maximum(mx.abs(qn_dot), max_val)
h_den = mx.add(h_den, eps)  # Add eps AFTER max
```

### Why This Works

- `exp(-m)` is the smallest "reasonable" value given current max state
- `max(|q·n|, exp(-m))` ensures denominator ≥ some tiny but non-zero value
- Adding eps AFTER max prevents bias when denominator is already large

### Lesson

**Epsilon is for numerical safety, not normalization.**

Add `eps` only:
1. After all other stabilization (max, abs, etc.)
2. To prevent literal division by zero
3. As small as possible (1e-6, not 1e-3)

**Never use eps to fix algorithmic bugs.** Fix the algorithm.

**Reference:** Canonical implementation line 428-430

## Problem 7: Metal Argument Limits

### Symptom

Kernel launch fails with:
```
Metal API error: argument count exceeds limit
```

### Root Cause

Metal has hard limit: **31 buffers per kernel**.

Our kernel had:
- 8 input tensors
- 3 output tensors  
- 10 scalar parameters (each counts as buffer!)
- = 21 buffers (close to limit)

Adding one more param → crash.

### The Fix

**Pack scalars into struct:**
```python
# Instead of separate buffers:
# kernel(q, k, v, i, f, NH, S, DHQK, DHHV, NC, L, qk_scale, eps, ...)

# Pack into single buffer:
params = mx.array([NH, S, DHQK, DHHV, NC, L], dtype=mx.int64)
floats = mx.array([qk_scale, eps], dtype=mx.float32)
kernel(q, k, v, i, f, params, floats, ...)
```

Saved 8 buffer slots.

### Lesson

**Metal is not CUDA.**

- CUDA: 4096 args
- Metal: 31 args
- Plan for this from start

**When designing kernels:**
1. Pack scalars into arrays
2. Use uniforms for compile-time constants
3. Consider threadgroup memory limits (32KB, not 96KB)

**Reference:** `docs_archive/FIX_RMSNORM_METAL_KERNEL.md`

## Validation Strategy

### Layer-by-Layer Parity

**Don't just test end-to-end.** Compare every intermediate:

```python
# Bad test
pytorch_out = pytorch_model(input)
mlx_out = mlx_model(input)
assert mx.allclose(mlx_out, pytorch_out)  # Fails, no idea where

# Good test
for layer_idx in range(num_layers):
    pt_layer_out = pytorch_model.layers[layer_idx](input)
    mlx_layer_out = mlx_model.layers[layer_idx](input)
    
    assert mx.allclose(mlx_layer_out, pt_layer_out, rtol=1e-5), \
        f"Layer {layer_idx} diverged"
    
    input = mlx_layer_out  # Continue with MLX output
```

This pinpoints exactly which layer diverges.

### State Validation

**Check state tensors, not just outputs:**

```python
mlx_h, (mlx_C, mlx_n, mlx_m) = mlx_mlstm(x, state)
pt_h, (pt_C, pt_n, pt_m) = pt_mlstm(x, state)

assert mx.allclose(mlx_C, pt_C, rtol=1e-5), "C diverged"
assert mx.allclose(mlx_n, pt_n, rtol=1e-5), "n diverged"
assert mx.allclose(mlx_m, pt_m, rtol=1e-5), "m diverged"
assert mx.allclose(mlx_h, pt_h, rtol=1e-5), "output diverged"
```

State errors compound over time. Catch them early.

### Numerical Ranges

**Log min/max/mean before crashes:**

```python
def check_tensor(x, name):
    print(f"{name}: min={mx.min(x):.6f}, max={mx.max(x):.6f}, mean={mx.mean(x):.6f}")
    
    if mx.any(mx.isnan(x)):
        print(f"❌ {name} has NaNs!")
    if mx.any(mx.isinf(x)):
        print(f"❌ {name} has Infs!")
```

Run this at every layer during debugging. Shows where things go wrong.

### Tolerance Guidelines

**Relative tolerance by dtype:**
- Float32: `rtol=1e-5` (5 decimal places)
- BFloat16: `rtol=1e-2` (2 decimal places)
- Float16: `rtol=5e-3` (3 decimal places)

**Absolute tolerance for near-zero:**
- Use `atol=1e-8` when values can be exactly zero

```python
assert mx.allclose(a, b, rtol=1e-5, atol=1e-8)
```

## Debugging Checklist

When you hit NaNs:

1. **Check dtypes**
   - [ ] Model uses `torch_dtype` not `autocast_kernel_dtype`?
   - [ ] State tensors in float32?
   - [ ] No implicit upcasts to float64?

2. **Check operations**
   - [ ] Using logsigmoid, not log+sigmoid?
   - [ ] Max-stabilized exponentials?
   - [ ] Denominator has lower bound + eps?

3. **Check shapes**
   - [ ] Covariance is [B, NH, QK_DH, V_DH] not reversed?
   - [ ] Outer products are k⊗v not v⊗k?
   - [ ] Assertions on all state shapes?

4. **Check scalars**
   - [ ] No float() or int() casts?
   - [ ] No Python arithmetic operators?
   - [ ] Using mx.add/multiply/divide?

5. **Check FFTs (if applicable)**
   - [ ] Exactly one 1/n factor?
   - [ ] Test with parseval identity?
   - [ ] Compared with reference implementation?

## Tools

**emberlint.py** - Catches precision violations:
```bash
python tools/emberlint.py --verbose --precision-only xlstm_metal/
```

**embercoach.py** - Explains violations (less strict):
```bash
python quarantine/embercoach.py xlstm_metal/
```

**Profiler** - Find CPU stalls:
```bash
python -m mlx.utils.profiler generate.py --model xlstm_7b_model --prompt "test"
```

## Summary: Hard-Won Rules

1. **Three dtypes, three purposes** - torch_dtype for model, others for specific ops
2. **One FFT normalization** - Never on both forward and inverse
3. **Zero Python scalars** - Use mx.array() and mx.ops
4. **Shapes are semantics** - k⊗v ≠ v⊗k even if dimensions match
5. **logsigmoid, not log+sigmoid** - Prevent underflow
6. **Max-stabilize exponentials** - exp(x - max(x)) ≤ 1.0
7. **eps after stabilization** - max(|x|, tiny) + eps, not x + eps
8. **Test every layer** - Don't just check end-to-end
9. **Pack Metal arguments** - 31 buffer limit, not 4096
10. **Trust nothing, verify everything** - Parity tests are mandatory

---

Every item in this list represents a multi-day debugging session. Learn from my pain. Don't repeat these mistakes.
