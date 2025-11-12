# MLX Numerics and Dtype Rules (Critical for Stability and Parity)

This document captures the rules we follow when writing MLX code and Metal-backed kernels to avoid silent float64
upcasts, ensure deterministic behavior, and preserve numerical stability. These patterns are mandatory for xLSTM on MLX.

## Core Principles

- Always operate in float32 (or the tensor's dtype). MLX defaults to float32; introducing Python floats (float64) into
  expressions can silently upcast intermediates and hurt stability/perf.
- All scalar constants that participate in MLX ops must be MLX arrays of the correct dtype.
- Prefer math in MLX (mx.*) over Python math; use broadcasting-aware MLX forms.
- Keep state tensors (C, N, M) in float32. Casting selectively for reductions can be used, but cast back to the input
  dtype afterwards when appropriate.

## Do and Don’t

- Do: wrap scalars in MLX arrays
    - Before:
        - `y = -mx.log(1.0 + mx.exp(-x))`  # Python 1.0 is float64
    - After:
        - `one = mx.array(1.0, dtype=x.dtype)`
        - `y = -mx.log(one + mx.exp(-x))`

- Do: use MLX rsqrt for inverse sqrt
    - Before: `q_scaled = q * (D ** (-0.5))`
    - After:  `q_scaled = q * mx.rsqrt(mx.array(D, dtype=q.dtype))`

- Do: keep eps as an MLX scalar of correct dtype
    - Before: `h = num / (den + eps)`
    - After:  `eps_a = mx.array(eps, dtype=den.dtype); h = num / (den + eps_a)`

- Don’t: pass Python infinities to `mx.where`
    - Before: `mx.where(mask, x, -float('inf'))`
    - After (float32 sentinel):
        - `neg_inf = mx.array(-3.4028235e38, dtype=x.dtype)`
        - `mx.where(mask, x, neg_inf)`

- Don’t: mix Python arithmetic with MLX tensors when a pure-MLX alternative exists.

## Canonical xLSTM Gate Identities (Transformers)

- `vecB = cumsum(logsigmoid(F), dim=-1)`
- `vecA = vecB_last - vecB + vecI`
- `scaG = vecB_last`
- `m_next = max(scaG + m_prev, max(vecA))`
- `scaGbar = exp(scaG + m_prev - m_next)`
- `vecAbar = exp(vecA - m_next[..., None])`

State updates per chunk:

- `Kbar = K_chunk * vecAbar[..., None]`
- `C_next = scaGbar[..., None, None] * C_prev + (Kbar^T @ V_chunk)`
- `N_next = scaGbar[..., None] * N_prev + sum_L(Kbar^T)`

Single-step output:

- `q_scaled = q * rsqrt(d_qk)`
- `den = max(|q_scaled·n|, exp(-m)) + eps`
- `h = (q_scaled^T @ C) / den`

## Metal Kernel Parameters

- Passing floats to Metal kernels is OK via bit-packing; those values are not fed through MLX ops.
- Inside MLX codepaths, never rely on Python float literals.

## Implementation Patterns (examples from repo)

- Log-sigmoid:
    - `one = mx.array(1.0, dtype=x.dtype)`
    - `f_log = -mx.log(one + mx.exp(-x))`

- Query scaling:
    - `q_scaled = q * mx.rsqrt(mx.array(QK_DH, dtype=q.dtype))`

- Denominator epsilon:
    - `eps_a = mx.array(eps, dtype=den.dtype)`
    - `h = num / (den + eps_a)`

- Causal mask replacement value:
    - `neg_inf = mx.array(-3.4028235e38, dtype=logits.dtype)`
    - `logits = mx.where(mask, logits, neg_inf)`

## Testing and Linting

- Run targeted parity tests vs the canonical PyTorch/Transformers formulas for A/G/C/N/M.
- EmberLint: if `emberlint.py` is available in your environment, run it to flag Python-float usage in MLX paths (
  constants like `1.0`, `-float('inf')`, `eps` additions). Example:
    - `python emberlint.py --paths xlstm_metal/blocks/mlstm_mlx --fix` (adjust to your env)
- In absence of EmberLint, quick grep helpers:
  -
  `rg -n "mx\.log\(1\.0|\+\s*1\.0|/\s*30\.0|/\s*15\.0|\+\s*eps\b|\(QK_DH\s*\*\*\s*\(-0\.5\)\)" xlstm_metal/blocks/mlstm_mlx`

## Why this matters

- MLX is float32-centric; accidental float64 upcasts destabilize exponential gating and parity with canonical kernels.
- Keeping numerics consistent prevents NaNs/Infs and preserves throughput on Apple GPUs.

## References

- Transformers xLSTM canonical implementation: `transformers/models/xlstm/modeling_xlstm.py`
- MLX kernels and usage: `docs/MLX_Metal_Kernel_Guide.md`, `docs/MLX_IMPLEMENTATION_GUIDE.md`

