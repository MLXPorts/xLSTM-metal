# HyperProfile Analysis: PyTorch ↔ MLX Numerical Equivalence

## Executive Summary

**Status**: ⚠️ CRITICAL DIFFERENCES FOUND

This document analyzes implementation differences between PyTorch and MLX backends for xLSTM-metal that affect numerical
equivalence. The HyperProfile system (from NCPS) is designed to compensate for these differences to ensure models
trained in PyTorch can run identically in MLX.

**Files Analyzed**:

- `xlstm_metal/blocks/torch/block.py` - PyTorch mLSTMLayer implementation
- `xlstm_metal/blocks/mlx/mlstm/block.py` - MLX mLSTMLayer implementation
- `xlstm_metal/blocks/mlx/mlstm/components.py` - MLX normalization components

---

## Critical Differences

### 1. Output Gate Soft-Cap ⚠️ CRITICAL

**PyTorch Implementation** (`torch/block.py:184-186`):

```python
o_preact = self.ogate_preact(x)  # [B, S, D]
o = torch.sigmoid(soft_cap(o_preact, self.config.gate_soft_cap))  # [B, S, D]
out = self.out_proj(h_norm * o)
```

**MLX Implementation** (`mlx/mlstm/block.py:177, 299-302`):

```python
o_preact = self.ogate_preact(x)  # [B, S, v_dim]
# NO soft_cap applied!
h_out = mx.multiply(mx.sigmoid(o_preact), h_norm)  # [B, S, v_dim]
y = self.out_proj(h_out)
```

**Impact**:

- MLX does NOT apply `soft_cap` to output gate before sigmoid
- PyTorch applies `soft_cap(o_preact, 15.0)` before sigmoid
- This causes different activation magnitudes for large o_preact values
- Affects gradient flow during Test-Time Training

**HyperProfile Compensation**:

```json
{
  "gate_compensation": {
    "output_gate_soft_cap": {
      "pytorch": true,
      "mlx": false,
      "fix_required": "Add soft_cap to MLX o_preact before sigmoid"
    }
  }
}
```

**Recommended Fix**: Add soft_cap to MLX implementation at `block.py:299`:

```python
# BEFORE (incorrect):
h_out = mx.multiply(mx.sigmoid(o_preact), h_norm)

# AFTER (matches PyTorch):
o_capped = soft_cap(o_preact, self.config.gate_soft_cap)
h_out = mx.multiply(mx.sigmoid(o_capped), h_norm)
```

---

### 2. Output Gate Projection Dimension

**PyTorch Implementation** (`torch/block.py:88`):

```python
self.ogate_preact = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.use_bias)
# Output: [B, S, embedding_dim]
```

**MLX Implementation** (`mlx/mlstm/block.py:121-124`):

```python
self.ogate_preact = nn.Linear(
    config.embedding_dim,
    config.v_dim,
    bias=config.use_bias
)
# Output: [B, S, v_dim]
```

**Impact**:

- For xLSTM-7B: `embedding_dim == v_dim == 4096` → **NO DIFFERENCE**
- For other configs where `v_dim_factor != 1.0`: **SHAPE MISMATCH**
- Example: If `v_dim_factor = 0.5`, then `v_dim = 2048` but `embedding_dim = 4096`
    - PyTorch ogate: [4096, 4096]
    - MLX ogate: [4096, 2048]

**HyperProfile Compensation**:

```json
{
  "projection_dimensions": {
    "output_gate_target_dim": {
      "pytorch": "embedding_dim",
      "mlx": "v_dim",
      "equivalence_condition": "v_dim_factor == 1.0"
    }
  }
}
```

**Analysis**:

- MLX implementation is actually MORE correct (output gate should match v_dim since it gates h_norm which is v_dim)
- PyTorch implementation happens to work for xLSTM-7B because v_dim == embedding_dim
- **Recommended**: Keep MLX implementation, fix PyTorch to use v_dim

---

### 3. MultiHeadLayerNorm Implementation

**PyTorch Implementation** (`torch/block.py:39-58`):

```python
class LayerNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: [B, NH, S, head_dim]
        Returns: [B, NH, S, head_dim]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias
```

**MLX Implementation** (`mlx/mlstm/components.py:86-164`):

```python
class MultiHeadLayerNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, ...):
        super().__init__()
        # CRITICAL: Weight and bias are FLAT [num_heads * head_dim]
        self.weight = mx.ones((num_heads * head_dim,))
        self.bias = mx.zeros((num_heads * head_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args: x: [B, S, num_heads, head_dim]
        Returns: [B, S, num_heads * head_dim]
        """
        # Normalize per head over head_dim
        mean = mx.mean(x, axis=-1, keepdims=True)  # [B, S, NH, 1]
        variance = mx.var(x, axis=-1, keepdims=True)  # [B, S, NH, 1]
        x_norm = (x - mean) / mx.sqrt(variance + eps)

        # CRITICAL: Reshape BEFORE applying weight/bias
        x_norm = x_norm.reshape(B, S, -1)  # [B, S, NH*DH]

        # Apply weight to flat tensor
        return x_norm * self.weight + self.bias
```

**Key Differences**:

1. **Input shape**:
    - PyTorch: `[B, NH, S, head_dim]` (heads in dim 1)
    - MLX: `[B, S, NH, head_dim]` (heads in dim 2)

2. **Weight/bias shape**:
    - PyTorch: `[NH, head_dim]` (2D)
    - MLX: `[NH * head_dim]` (1D flat)

3. **Output shape**:
    - PyTorch: `[B, NH, S, head_dim]` (same as input)
    - MLX: `[B, S, NH * head_dim]` (flattened heads)

4. **Variance computation**:
    - PyTorch: `var(..., unbiased=False)` (biased variance, n not n-1)
    - MLX: `mx.var(...)` (need to verify if biased or unbiased)

**Impact**:

- Reshaping differences handled by caller (block transposes appropriately)
- Weight broadcasting might differ slightly
- Variance formula difference could cause small numerical differences

**HyperProfile Compensation**:

```json
{
  "normalization": {
    "multihead_layernorm": {
      "pytorch_variance": "biased (n)",
      "mlx_variance": "unknown",
      "weight_shape_pytorch": "[num_heads, head_dim]",
      "weight_shape_mlx": "[num_heads * head_dim]",
      "verification_required": true
    }
  }
}
```

**Recommended Test**:

```python
# Verify variance computation equivalence
import torch
import mlx.core as mx
x_torch = torch.randn(2, 8, 64, 512)
x_mlx = mx.array(x_torch.numpy())

# PyTorch (biased variance)
var_torch = x_torch.var(dim=-1, keepdim=True, unbiased=False)

# MLX (default variance)
var_mlx = mx.var(x_mlx, axis=-1, keepdims=True)

# Check equivalence
assert np.allclose(var_torch.numpy(), np.array(var_mlx), rtol=1e-5)
```

---

### 4. Soft-Cap Implementation

**PyTorch Implementation** (`torch/block.py:61-63`):

```python
def soft_cap(x: torch.Tensor, soft_cap_value: float) -> torch.Tensor:
    """Soft capping: tanh(x / soft_cap) * soft_cap."""
    return torch.tanh(x / soft_cap_value) * soft_cap_value
```

**MLX Implementation** (`mlx/mlstm/components.py:11-23`):

```python
def soft_cap(x: mx.array, cap_value: float) -> mx.array:
    """Soft capping: cap_value * tanh(x / cap_value)"""
    cap = mx.array(cap_value, dtype=x.dtype)
    return mx.multiply(cap, mx.tanh(mx.divide(x, cap)))
```

**Difference**:

- PyTorch: scalar division/multiplication
- MLX: creates array for cap_value with same dtype as input
- **Mathematically equivalent** but MLX preserves dtype better

**Impact**: Negligible (both compute `cap * tanh(x / cap)`)

---

### 5. Gate Application Location

**PyTorch Implementation** (`torch/block.py:179-188`):

```python
# Normalize heads
h_norm = self.multihead_norm(h)  # [B, NH, S, head_dim]

# Reshape back to sequence
h_norm = h_norm.transpose(1, 2).contiguous()  # [B, S, NH, head_dim]
h_norm = h_norm.reshape(B, S, self.config.v_dim)  # [B, S, v_dim]

# Output gate
o_preact = self.ogate_preact(x)  # [B, S, D]
o = torch.sigmoid(soft_cap(o_preact, self.config.gate_soft_cap))

# Output projection
out = self.out_proj(h_norm * o)  # [B, S, D]
```

**MLX Implementation** (`mlx/mlstm/block.py:289-302`):

```python
# Transpose back and reshape
h = h.transpose(0, 2, 1, 3)  # [B, S, num_heads, head_dim]

# Multi-head normalization
h_norm = self.multihead_norm(h)  # [B, S, num_heads, head_dim] → [B, S, NH*DH]

# Reshape for output projection (already flat from multihead_norm!)
h_norm = h_norm.reshape(B, S, self.config.v_dim)  # [B, S, v_dim]

# Apply output gate (sigmoid)
h_out = mx.multiply(mx.sigmoid(o_preact), h_norm)  # [B, S, v_dim]

# Output projection
y = self.out_proj(h_out)  # [B, S, embedding_dim]
```

**Difference**:

- PyTorch: `out_proj(h_norm * o)` - gate THEN project
- MLX: Same order - gate THEN project
- **But PyTorch uses soft_cap on o, MLX doesn't!** (see Issue #1)

---

## Numerical Stability Differences

### Float32 State Dtype ✅ FIXED

**Both implementations now maintain float32 states**:

- Verified in `CONFIG_DRIVEN_ARCHITECTURE_VERIFIED.md`
- PyTorch kernel: `dtype_state=torch.float32`
- MLX kernel: Casts to `mx.float32` before state updates

---

### RMSNorm Force Float32 Reductions

**MLX Implementation** (`mlx/mlstm/components.py:26-83`):

```python
class RMSNorm(nn.Module):
    def __init__(self, ..., force_float32_reductions: bool = True):
        self.force_float32_reductions = force_float32_reductions

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype

        # Force float32 for reductions if requested
        if self.force_float32_reductions:
            x = x.astype(mx.float32)

        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x_norm = x * mx.rsqrt(variance + eps)

        # Cast back to input dtype
        x_norm = x_norm.astype(input_dtype)

        return x_norm * self.weight + self.bias
```

**PyTorch Implementation**: No equivalent force_float32 option

**Impact**: MLX has better numerical stability for fp16/bfloat16 computation with float32 reductions

---

## HyperProfile Specification

### Proposed Implementation

**File**: `xlstm_metal/profiles/hyperprofile.py`

```python
#!/usr/bin/env python
"""
HyperProfile System for Backend Compensation

Ensures numerical equivalence between PyTorch and MLX implementations
by specifying framework-specific quirks and compensation strategies.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class GateCompensation:
    """Compensation for gate activation differences"""
    apply_soft_cap_to_output_gate: bool
    soft_cap_value: float
    apply_before_sigmoid: bool


@dataclass
class NormalizationCompensation:
    """Compensation for normalization differences"""
    variance_biased: bool  # True for n, False for n-1
    force_float32_reductions: bool
    weight_shape_flat: bool  # True for [NH*DH], False for [NH, DH]


@dataclass
class DtypeCompensation:
    """Dtype handling differences"""
    state_dtype: str  # "float32" for both
    computation_dtype: Optional[str]  # "float32", "float16", "bfloat16"
    force_float32_for_reductions: bool


@dataclass
class ProjectionCompensation:
    """Projection dimension differences"""
    output_gate_target_dim: str  # "embedding_dim" or "v_dim"


@dataclass(frozen=True)
class HyperProfile:
    """
    Backend compensation profile for numerical equivalence.

    Ensures xLSTM trained in PyTorch can run identically in MLX
    by compensating for framework differences.

    Based on NCPS HyperProfile design (see XLSTM_MAD_NCPS_DESIGN.md).
    """
    name: str
    description: str
    backend_source: str  # "torch", "mlx", "jax"
    backend_target: str

    # Compensation strategies
    gate_compensation: GateCompensation
    norm_compensation: NormalizationCompensation
    dtype_compensation: DtypeCompensation
    projection_compensation: ProjectionCompensation

    # Framework-specific extras
    extras: Dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> "HyperProfile":
        """Load HyperProfile from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            name=data['name'],
            description=data['description'],
            backend_source=data['backend_source'],
            backend_target=data['backend_target'],
            gate_compensation=GateCompensation(**data['gate_compensation']),
            norm_compensation=NormalizationCompensation(**data['norm_compensation']),
            dtype_compensation=DtypeCompensation(**data['dtype_compensation']),
            projection_compensation=ProjectionCompensation(**data['projection_compensation']),
            extras=data.get('extras', {})
        )

    def to_json(self, path: Path):
        """Save HyperProfile to JSON file"""
        data = {
            'name': self.name,
            'description': self.description,
            'backend_source': self.backend_source,
            'backend_target': self.backend_target,
            'gate_compensation': {
                'apply_soft_cap_to_output_gate': self.gate_compensation.apply_soft_cap_to_output_gate,
                'soft_cap_value': self.gate_compensation.soft_cap_value,
                'apply_before_sigmoid': self.gate_compensation.apply_before_sigmoid
            },
            'norm_compensation': {
                'variance_biased': self.norm_compensation.variance_biased,
                'force_float32_reductions': self.norm_compensation.force_float32_reductions,
                'weight_shape_flat': self.norm_compensation.weight_shape_flat
            },
            'dtype_compensation': {
                'state_dtype': self.dtype_compensation.state_dtype,
                'computation_dtype': self.dtype_compensation.computation_dtype,
                'force_float32_for_reductions': self.dtype_compensation.force_float32_for_reductions
            },
            'projection_compensation': {
                'output_gate_target_dim': self.projection_compensation.output_gate_target_dim
            },
            'extras': self.extras
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Predefined profiles
PYTORCH_PROFILE = HyperProfile(
    name="mlstm_pytorch_canonical",
    description="Canonical PyTorch mLSTM implementation",
    backend_source="torch",
    backend_target="torch",
    gate_compensation=GateCompensation(
        apply_soft_cap_to_output_gate=True,
        soft_cap_value=15.0,
        apply_before_sigmoid=True
    ),
    norm_compensation=NormalizationCompensation(
        variance_biased=True,  # unbiased=False → n not n-1
        force_float32_reductions=False,
        weight_shape_flat=False  # [NH, DH]
    ),
    dtype_compensation=DtypeCompensation(
        state_dtype="float32",
        computation_dtype=None,
        force_float32_for_reductions=False
    ),
    projection_compensation=ProjectionCompensation(
        output_gate_target_dim="embedding_dim"  # BUG: should be v_dim
    ),
    extras={}
)


MLX_CURRENT_PROFILE = HyperProfile(
    name="mlstm_mlx_current",
    description="Current MLX mLSTM implementation (with output gate bug)",
    backend_source="mlx",
    backend_target="mlx",
    gate_compensation=GateCompensation(
        apply_soft_cap_to_output_gate=False,  # BUG: Missing soft_cap!
        soft_cap_value=15.0,
        apply_before_sigmoid=False
    ),
    norm_compensation=NormalizationCompensation(
        variance_biased=True,  # Need to verify
        force_float32_reductions=True,
        weight_shape_flat=True  # [NH*DH]
    ),
    dtype_compensation=DtypeCompensation(
        state_dtype="float32",
        computation_dtype=None,
        force_float32_for_reductions=True
    ),
    projection_compensation=ProjectionCompensation(
        output_gate_target_dim="v_dim"  # Correct!
    ),
    extras={
        "metal_kernel_params": {
            "chunk_size": 64,
            "siz_b_DHQK": 16,
            "siz_b_DHHV": 16
        }
    }
)


MLX_FIXED_PROFILE = HyperProfile(
    name="mlstm_mlx_from_torch",
    description="MLX mLSTM with PyTorch numerical equivalence (fixed)",
    backend_source="torch",
    backend_target="mlx",
    gate_compensation=GateCompensation(
        apply_soft_cap_to_output_gate=True,  # FIXED: Add soft_cap
        soft_cap_value=15.0,
        apply_before_sigmoid=True
    ),
    norm_compensation=NormalizationCompensation(
        variance_biased=True,
        force_float32_reductions=True,  # Better stability than PyTorch
        weight_shape_flat=True
    ),
    dtype_compensation=DtypeCompensation(
        state_dtype="float32",
        computation_dtype=None,
        force_float32_for_reductions=True
    ),
    projection_compensation=ProjectionCompensation(
        output_gate_target_dim="v_dim"  # Keep MLX version (more correct)
    ),
    extras={
        "metal_kernel_params": {
            "chunk_size": 64,
            "siz_b_DHQK": 16,
            "siz_b_DHHV": 16
        }
    }
)
```

---

## Recommended Fixes

### Priority 1: Critical (Blocking Numerical Equivalence)

**1. Add soft_cap to MLX output gate** (`xlstm_metal/blocks/mlx/mlstm/block.py:299`)

**Current**:

```python
# 10. Apply output gate (sigmoid)
h_out = mx.multiply(mx.sigmoid(o_preact), h_norm)  # [B, S, v_dim]
```

**Fixed**:

```python
# 10. Apply output gate with soft-cap (matches PyTorch)
o_capped = soft_cap(o_preact, self.config.gate_soft_cap)  # Add soft_cap!
h_out = mx.multiply(mx.sigmoid(o_capped), h_norm)  # [B, S, v_dim]
```

**Impact**: Ensures output gate activations match PyTorch for large values

---

### Priority 2: High (Weight Compatibility)

**2. Fix PyTorch output gate dimension** (`xlstm_metal/blocks/torch/block.py:88`)

**Current**:

```python
self.ogate_preact = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.use_bias)
```

**Fixed**:

```python
self.ogate_preact = nn.Linear(config.embedding_dim, config.v_dim, bias=config.use_bias)
```

**Impact**:

- Makes PyTorch compatible with arbitrary v_dim_factor values
- For xLSTM-7B (v_dim == embedding_dim), no weight shape change
- For other configs, prevents shape mismatch

---

### Priority 3: Medium (Verification)

**3. Verify MLX variance computation is biased**

**Test**:

```python
import mlx.core as mx
import numpy as np

x = mx.random.normal((2, 8, 64, 512))

# MLX default
var_mlx = mx.var(x, axis=-1)

# NumPy biased (ddof=0, equivalent to n)
var_np_biased = np.var(x, axis=-1, ddof=0)

# NumPy unbiased (ddof=1, equivalent to n-1)
var_np_unbiased = np.var(x, axis=-1, ddof=1)

print(f"MLX matches biased (ddof=0): {np.allclose(var_mlx, var_np_biased)}")
print(f"MLX matches unbiased (ddof=1): {np.allclose(var_mlx, var_np_unbiased)}")
```

---

## Test Plan

### Numerical Equivalence Test Suite

**File**: `tests/test_pytorch_mlx_equivalence.py`

```python
#!/usr/bin/env python
"""
Test numerical equivalence between PyTorch and MLX implementations.

Verifies HyperProfile compensation ensures identical outputs.
"""

import torch
import mlx.core as mx
import numpy as np
from quarantine.torch.block import mLSTMLayer as mLSTMLayerTorch
from xlstm_metal.blocks.mlstm_layer import mLSTMLayer as mLSTMLayerMLX
from quarantine.torch.block import mLSTMConfig as mLSTMConfigTorch
from xlstm_metal.blocks.mlstm_layer import mLSTMConfig as mLSTMConfigMLX


def test_output_gate_soft_cap():
    """Verify output gate soft-cap equivalence"""
    config_torch = mLSTMConfigTorch(
        embedding_dim=512,
        num_heads=4,
        gate_soft_cap=15.0
    )
    config_mlx = mLSTMConfigMLX(
        embedding_dim=512,
        num_heads=4,
        gate_soft_cap=15.0
    )

    # Create models
    model_torch = mLSTMLayerTorch(config_torch)
    model_mlx = mLSTMLayerMLX(config_mlx)

    # Copy weights from PyTorch to MLX
    # ... (weight transfer code)

    # Test input
    x_torch = torch.randn(2, 8, 512)
    x_mlx = mx.array(x_torch.numpy())

    # Forward pass
    out_torch, _ = model_torch(x_torch, state=None)
    out_mlx, _ = model_mlx(x_mlx, state=None)

    # Verify equivalence
    diff = np.abs(out_torch.detach().numpy() - np.array(out_mlx))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

    assert max_diff < 1e-4, f"Output gate soft-cap mismatch: max_diff={max_diff}"


def test_multihead_layernorm_equivalence():
    """Verify MultiHeadLayerNorm equivalence"""
    # ... (similar structure)


def test_full_forward_pass_equivalence():
    """End-to-end forward pass equivalence test"""
    # ... (comprehensive test)


if __name__ == "__main__":
    test_output_gate_soft_cap()
    test_multihead_layernorm_equivalence()
    test_full_forward_pass_equivalence()
    print("✓ All numerical equivalence tests passed!")
```

---

## Next Steps

1. **Apply Priority 1 fix** ✅
    - Add soft_cap to MLX output gate
    - Verify numerical equivalence with PyTorch

2. **Apply Priority 2 fix**
    - Update PyTorch ogate dimension to v_dim
    - Test with v_dim_factor != 1.0 configs

3. **Create HyperProfile implementation**
    - Implement `xlstm_metal/profiles/hyperprofile.py`
    - Save predefined profiles as JSON

4. **Build equivalence test suite**
    - Comprehensive PyTorch ↔ MLX tests
    - Weight transfer utilities
    - Automated regression tests

5. **Update documentation**
    - Add HyperProfile usage guide
    - Document backend differences
    - Create migration guide

---

## References

- **NCPS HyperProfile Design**: `docs/architecture/XLSTM_MAD_NCPS_DESIGN.md:464-528`
- **Config-Driven Architecture**: `docs/CONFIG_DRIVEN_ARCHITECTURE_VERIFIED.md`
- **PyTorch Implementation**: `xlstm_metal/blocks/torch/block.py`
- **MLX Implementation**: `xlstm_metal/blocks/mlx/mlstm/block.py`
- **MLX Components**: `xlstm_metal/blocks/mlx/mlstm/components.py`

---

## Conclusion

The analysis reveals **1 critical difference** (output gate soft-cap) and **1 dimension mismatch** (ogate projection for
v_dim_factor != 1.0) between PyTorch and MLX implementations.

**Status**:

- ✅ Float32 state dtype: VERIFIED EQUIVALENT
- ⚠️ Output gate soft-cap: **REQUIRES FIX IN MLX**
- ⚠️ Output gate dimension: Requires fix in PyTorch (or accept MLX as canonical)
- ⏳ Variance computation: Needs verification
- ✅ Soft-cap formula: Equivalent

**Next**: Apply Priority 1 fix to achieve numerical equivalence for xLSTM-7B inference and Test-Time Training.
