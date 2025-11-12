# mLSTM Numerical Stability Analysis

**Date:** 2025-01-29  
**Status:** Investigation - Numerical Instability in MLX Inference  
**Context:** Recent numerical instability introduced in MLX inference path

## Problem Statement

Numerical instability has been observed in the MLX inference path for mLSTM. This document analyzes the canonical
transformers implementation to identify critical numerical stability patterns and compares them with our MLX
implementation.

## Canonical Implementation (transformers/modeling_xlstm.py)

### Critical Numerical Stability Patterns

#### 1. **Forget Gate: logsigmoid BEFORE exponential**

```python
# CANONICAL (transformers line ~260):
scaF_log = torch.nn.functional.logsigmoid(fgate)
# Then use scaF_log in exponential calculations

# NOT: scaF = sigmoid(fgate) then log(scaF)
# The logsigmoid is more numerically stable!
```

**Why this matters:**

- `logsigmoid(x) = -log(1 + exp(-x))` is stable for large negative x
- `log(sigmoid(x))` can underflow when `sigmoid(x) ≈ 0`

**Our MLX Implementation** (kernel.py line 60):

```python
# ✅ CORRECT - matches canonical
f_log = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_preact)))))  # logsigmoid
```

#### 2. **Max State Update: Exact Order Matters**

```python
# CANONICAL (transformers line ~413):
scaM_state_new = torch.max(scaF_log + scaM_old, igate)

# Compute normalized gates:
scaF_act = torch.exp(scaF_log + scaM_old - scaM_state_new)
scaI_act = torch.exp(igate - scaM_state_new)
```

**Critical insight:**

- Max state prevents exponential overflow
- All exponentials are computed relative to `scaM_state_new`
- This ensures `exp(x - scaM_state_new)` is always ≤ 1.0

**Our MLX Implementation** (kernel.py lines 65-70):

```python
# ✅ CORRECT - matches canonical
m_new = mx.maximum(mx.add(f_log, m_state), i_preact)

f_exp = mx.exp(mx.subtract(mx.add(f_log, m_state), m_new))
i_exp = mx.exp(mx.subtract(i_preact, m_new))
```

#### 3. **Query Scaling: Applied DURING Retrieval**

```python
# CANONICAL (transformers line ~419):
vecQ_scaled = query * (dhqk ** (-0.5))

# Use scaled query for:
# 1. Computing numerator
# 2. Computing denominator dot product
h_num = vecQ_scaled[:, :, None, :] @ matC_state_new
qn_dotproduct = vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None]
```

**Critical:** Key is NOT scaled when storing! Scaling happens on query side only.

**Our MLX Implementation** (kernel.py lines 91-92):

```python
# ✅ CORRECT - matches canonical
q_scaled = mx.multiply(q, mx.rsqrt(mx.array(QK_DH, dtype=q.dtype)))
```

#### 4. **Denominator Stabilization: max(|q·n|, exp(-m)) + eps**

```python
# CANONICAL (transformers lines ~428-430):
max_val = torch.exp(-scaM_state_new)
h_denom = (torch.maximum(qn_dotproduct.abs(), max_val) + eps)
h = h_num / h_denom
```

**Critical numerical pattern:**

- Use **absolute value** of q·n dot product
- Lower bound by `exp(-m)` (never goes to zero)
- Add eps only after max operation
- Division is safe because denominator ≥ `exp(-m) + eps`

**Our MLX Implementation** (kernel.py lines 96-101):

```python
# ✅ CORRECT - matches canonical
qn_dot = mx.sum(mx.multiply(n_new, q_scaled), axis=-1, keepdims=True)
max_val = mx.exp(mx.negative(m_new))[:, :, None]
eps_a = mx.array(eps, dtype=q.dtype)
h_den = mx.add(mx.maximum(mx.abs(qn_dot), max_val), eps_a)
h = mx.divide(h_num, h_den)
```

#### 5. **State Dtype Management**

```python
# CANONICAL (transformers line ~408):
dtype_state: torch.dtype = torch.float32  # State is ALWAYS float32

# Convert to computation dtype only when needed:
matC_state_new = scaF_act[:, :, :, None] * matC_old + ...
h_num = vecQ_scaled[:, :, None, :] @ matC_state_new.to(dtype=dtype_qkv)  # ← convert here

# Convert back to float32 for state storage:
matC_state_new = matC_state_new.to(dtype=dtype_state)
```

**Critical insight:**

- States (C, n, m) stored in float32 always
- Computation can be bfloat16/float16
- Cast to computation dtype only for matmul
- Cast back to float32 for state updates

**Potential Issue in Our MLX Implementation:**
We may not be properly managing dtype conversions!

#### 6. **Covariance Update: Shape and Order**

```python
# CANONICAL (transformers line ~421):
matC_state_new = scaF_act[:, :, :, None] * matC_old + scaI_act[:, :, :, None] * (
    key[:, :, :, None] @ value[:, :, None, :]
)
```

**Shape verification:**

- `key`: [B, NH, DHQK]
- `value`: [B, NH, DHV]
- `key[:, :, :, None]`: [B, NH, DHQK, 1]
- `value[:, :, None, :]`: [B, NH, 1, DHV]
- Outer product: [B, NH, DHQK, DHV]

This is `k ⊗ v`, NOT `v ⊗ k`!

**Our MLX Implementation** (kernel.py lines 79-83):

```python
# ✅ CORRECT - matches canonical
k_expanded = k[:, :, :, None]  # [B, NH, QK_DH, 1]
v_expanded = v[:, :, None, :]  # [B, NH, 1, V_DH]
kv_outer = mx.multiply(k_expanded, v_expanded)  # [B, NH, QK_DH, V_DH]
c_new = mx.add(mx.multiply(f_expanded, c_state), mx.multiply(i_expanded, kv_outer))
```

## Potential Issues in Our MLX Implementation

### Issue 1: Dtype Management (HIGH PRIORITY)

**Problem:** We may not be properly maintaining float32 for states.

**Check locations:**

1. `xlstm_metal/blocks/mlx/mlstm/block.py` - mLSTMLayer initialization
2. State initialization in `mlstm_sequential` and `mlstm_chunkwise`
3. Gate computation dtype handling

**Fix needed:**

```python
# In state initialization:
if c_initial is None:
    c_state = mx.zeros((B, NH, QK_DH, V_DH), dtype=mx.float32)  # ← Force float32!
else:
    c_state = c_initial.astype(mx.float32)  # ← Cast to float32!

# In computation:
# Cast to computation dtype for matmul
h_num = mx.matmul(c_new.astype(q.dtype), q_scaled[..., None])

# Cast back to float32 for state storage
c_new = c_new.astype(mx.float32)
```

### Issue 2: Epsilon Placement

**Current:**

```python
# kernel.py line 101:
h_den = mx.add(mx.maximum(mx.abs(qn_dot), max_val), eps_a)
```

**Verify:**

- Is `eps` the same value as canonical? (canonical uses 1e-6)
- Is `eps_a` properly typed to match computation dtype?

### Issue 3: Gate Soft-Capping

**From config.json:**

```json
"gate_soft_cap": 15.0
```

**Check:** Are we applying soft-cap to gates BEFORE the mlstm kernel?

**Expected:** In mLSTMLayer, BEFORE calling kernel:

```python
# Apply soft-cap to gates (config.json line 25)
if self.config.gate_soft_cap:
    cap = self.config.gate_soft_cap
    i_preact = cap * mx.tanh(i_preact / cap)
    f_preact = cap * mx.tanh(f_preact / cap)
```

### Issue 4: Sequence Length Handling

**Canonical wraps with padding:**

```python
# transformers line ~560:
def wrap_chunkwise_pad_zeros(...):
    if sequence_length % chunk_size != 0:
        S_padded = ((sequence_length + chunk_size - 1) // chunk_size) * chunk_size
        # Pad with zeros...
```

**Check:** Do we properly handle sequences not divisible by chunk_size?

## HyperProfile Implications

Based on NCPS hyperprofile pattern, we should create:

```json
// xlstm_metal/profiles/mlstm_inference_mlx_from_torch.json
{
  "name": "mlstm_inference_mlx_from_torch",
  "description": "mLSTM inference on MLX matching PyTorch Transformers numerics",
  "backend_source": "torch",
  "backend_target": "mlx",
  
  "initializers": {
    "exponential_gate": {
      "weight_init": "normal",
      "weight_std": 0.02,
      "bias_init_range_igate": [-10.0, -10.0],
      "bias_init_range_fgate": [0.0, 0.0]
    },
    "matrix_memory": {
      "init_scale": 0.0
    }
  },
  
  "constraints": {
    "exponential_gate": {
      "soft_cap": 15.0,
      "preact_clamp_min": -50.0,
      "preact_clamp_max": 50.0
    },
    "denominator": {
      "eps": 1e-6,
      "min_exp_m": 1e-30
    },
    "state_dtype": {
      "c_state": "float32",
      "n_state": "float32",
      "m_state": "float32",
      "computation_dtype": "float32"
    }
  },
  
  "solver_config": {
    "chunk_size": 64,
    "use_logsigmoid": true,
    "scale_query_not_key": true,
    "use_abs_in_denominator": true
  }
}
```

## Testing Strategy (GPT-5 Approach)

Create deterministic numerical tests:

```python
# tests/test_mlx_torch_numerical_parity.py
import mlx.core as mx
import torch
import numpy as np

def test_logsigmoid_parity():
    """Test logsigmoid numerical parity."""
    # Deterministic test values
    x_np = np.array([-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0], dtype=np.float32)
    
    # PyTorch reference
    x_torch = torch.from_numpy(x_np)
    result_torch = torch.nn.functional.logsigmoid(x_torch).numpy()
    
    # MLX implementation
    x_mlx = mx.array(x_np)
    result_mlx = mx.negative(mx.log(mx.add(1.0, mx.exp(mx.negative(x_mlx)))))
    result_mlx = np.array(result_mlx)
    
    # Compare
    np.testing.assert_allclose(result_mlx, result_torch, rtol=1e-6, atol=1e-8)

def test_exponential_gating_parity():
    """Test exponential gating numerical stability."""
    # Edge cases for exponential gating
    i_preact = np.array([[-10.0], [0.0], [10.0]], dtype=np.float32)
    f_preact = np.array([[-10.0], [0.0], [10.0]], dtype=np.float32)
    m_state = np.array([[0.0], [5.0], [-5.0]], dtype=np.float32)
    
    # PyTorch reference (from transformers)
    i_torch = torch.from_numpy(i_preact)
    f_torch = torch.from_numpy(f_preact)
    m_torch = torch.from_numpy(m_state)
    
    f_log_torch = torch.nn.functional.logsigmoid(f_torch)
    m_new_torch = torch.max(f_log_torch + m_torch, i_torch)
    f_exp_torch = torch.exp(f_log_torch + m_torch - m_new_torch)
    i_exp_torch = torch.exp(i_torch - m_new_torch)
    
    # MLX implementation
    i_mlx = mx.array(i_preact)
    f_mlx = mx.array(f_preact)
    m_mlx = mx.array(m_state)
    
    f_log_mlx = mx.negative(mx.log(mx.add(1.0, mx.exp(mx.negative(f_mlx)))))
    m_new_mlx = mx.maximum(mx.add(f_log_mlx, m_mlx), i_mlx)
    f_exp_mlx = mx.exp(mx.subtract(mx.add(f_log_mlx, m_mlx), m_new_mlx))
    i_exp_mlx = mx.exp(mx.subtract(i_mlx, m_new_mlx))
    
    # Compare
    np.testing.assert_allclose(np.array(m_new_mlx), m_new_torch.numpy(), rtol=1e-6)
    np.testing.assert_allclose(np.array(f_exp_mlx), f_exp_torch.numpy(), rtol=1e-6)
    np.testing.assert_allclose(np.array(i_exp_mlx), i_exp_torch.numpy(), rtol=1e-6)

def test_denominator_stability():
    """Test denominator computation with edge cases."""
    # Create scenario where |q·n| is very small
    q_scaled = np.array([[[1e-8, 1e-8, 1e-8]]], dtype=np.float32)  # [1, 1, 3]
    n_state = np.array([[[1e-8, -1e-8, 0.0]]], dtype=np.float32)  # [1, 1, 3]
    m_state = np.array([[[10.0]]], dtype=np.float32)  # [1, 1, 1]
    eps = 1e-6
    
    # PyTorch reference
    q_torch = torch.from_numpy(q_scaled)
    n_torch = torch.from_numpy(n_state)
    m_torch = torch.from_numpy(m_state)
    
    qn_dot_torch = (q_torch * n_torch).sum(dim=-1, keepdim=True)
    max_val_torch = torch.exp(-m_torch)
    denom_torch = torch.maximum(qn_dot_torch.abs(), max_val_torch) + eps
    
    # MLX implementation
    q_mlx = mx.array(q_scaled)
    n_mlx = mx.array(n_state)
    m_mlx = mx.array(m_state)
    
    qn_dot_mlx = mx.sum(mx.multiply(q_mlx, n_mlx), axis=-1, keepdims=True)
    max_val_mlx = mx.exp(mx.negative(m_mlx))
    denom_mlx = mx.add(mx.maximum(mx.abs(qn_dot_mlx), max_val_mlx), eps)
    
    # Compare
    np.testing.assert_allclose(np.array(denom_mlx), denom_torch.numpy(), rtol=1e-6)
    
    # Verify denominator is never zero or too small
    assert np.all(np.array(denom_mlx) > eps)
```

## Action Items

1. **Immediate: Dtype Management**
    - [ ] Audit all state initialization for float32
    - [ ] Add explicit dtype casts in recurrent step
    - [ ] Test with mixed precision (bfloat16 computation, float32 states)

2. **Immediate: Gate Soft-Capping**
    - [ ] Verify soft-cap is applied in mLSTMLayer.__call__
    - [ ] Check config.gate_soft_cap value matches canonical (15.0)

3. **High Priority: Numerical Parity Tests**
    - [ ] Implement deterministic tests (above)
    - [ ] Test edge cases (large pos/neg, near-zero)
    - [ ] Compare MLX vs PyTorch at every intermediate step

4. **Medium Priority: HyperProfile**
    - [ ] Create MLX-from-Torch profile
    - [ ] Load profile in mLSTMLayer initialization
    - [ ] Apply constraints and initializers from profile

5. **Documentation**
    - [ ] Document numerical stability guarantees
    - [ ] Add comments explaining each stabilization technique
    - [ ] Create debugging guide for numerical issues

## References

- Transformers xLSTM:
  `<local_install>/site-packages/transformers/models/xlstm/modeling_xlstm.py`
- Our MLX kernel: `xlstm_metal/blocks/mlx/mlstm/kernel.py`
- Config: `xlstm_7b_model/config.json`
- Canonical notes: `docs/porting/CANONICAL_XLSTM_IMPLEMENTATION_NOTES.md`
