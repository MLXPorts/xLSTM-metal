"""Multi-Head Normalization Layers – MLX Implementation

Overview
--------
Multi-head normalization applies layer normalization or RMS normalization
**independently per head** rather than across the entire flattened dimension.
This is critical for multi-head architectures (like mLSTM) where different
heads may operate at different activation scales.

Why Per-Head Normalization?
----------------------------
In standard LayerNorm/RMSNorm over a flattened [NH * DH] dimension, the
normalization statistics (mean, variance, RMS) are computed globally. This
can be problematic when:
  - One head dominates in magnitude (its stats swamp others)
  - Different heads encode different types of information at different scales
  - Gradient flow becomes imbalanced across heads

Per-head normalization computes stats independently for each head's DH
features, ensuring each head is normalized to a consistent scale before
being combined.

Tensor Flow
-----------
Input:  [B, S, NH, DH]  (multi-head format)
  ↓ Normalize each [:, :, h, :] independently
Normalized: [B, S, NH, DH]
  ↓ Flatten to [B, S, NH * DH]
Output: [B, S, NH * DH]  (ready for projection)

Weight Shape
------------
**CRITICAL**: Weight is stored as a **flat vector [NH * DH]**, not [NH, DH].
This matches the HuggingFace transformers xLSTM implementation and allows
a single learnable parameter per feature dimension.

After per-head normalization and flattening, the weight is applied:
  output = normalized_flat * weight

This design enables the model to learn per-feature rescaling while maintaining
per-head normalization statistics.

MultiHeadLayerNorm vs MultiHeadRMSNorm
--------------------------------------
- **LayerNorm**: Computes mean and variance per head, normalizes via
  (x - mean) / sqrt(var + eps). More compute (two passes) but zero-centers.

- **RMSNorm**: Computes only RMS = sqrt(mean(x²) + eps), normalizes via
  x / RMS. Single pass, no mean centering. Often equivalent performance.

Force Float32 Reductions
------------------------
When `force_float32_reductions=True`, mean/variance/RMS computations are
performed in float32 even if inputs are bfloat16. This prevents accumulation
errors in long reductions (large DH) and is **strongly recommended** for
stable mixed-precision training.

Usage in mLSTM
--------------
The mLSTM output cell uses MultiHeadRMSNorm to normalize hidden states h
before applying the output gate and projection:

  h: [B, NH, S, DH_v] → transpose → [B, S, NH, DH_v]
  h_norm = MultiHeadRMSNorm(h)  → [B, S, NH * DH_v]
  output = LinearProjection(h_norm * output_gate)

This ensures each head's contribution is properly scaled before the final
projection back to embedding_dim.

Parity
------
Logic mirrors torch-native MultiHeadLayerNorm/MultiHeadRMSNorm for testing.
"""

import mlx.core as mx
import mlx.nn as nn


class MultiHeadLayerNorm(nn.Module):
    """Per-head LayerNorm with flattening (mean centering + variance scaling).

    Applies standard LayerNorm independently to each head's features, then
    flattens and applies a shared weight vector. Commonly used when zero-
    centering is beneficial for downstream layers.

    Parameters
    ----------
    num_heads : int
        Number of attention heads (NH).
    head_dim : int
        Dimension per head (DH).
    eps : float, default 1e-6
        Numerical stability epsilon for variance.
    use_weight : bool, default True
        Whether to apply learnable weight scaling.
    use_bias : bool, default False
        Whether to apply learnable bias (after normalization).

    Returns (forward)
    -----------------
    output : mx.array [B, S, NH * DH]
        Normalized and flattened activations.

    Notes
    -----
    Weight/bias are flat [NH * DH], applied **after** per-head normalization
    and flattening (matches HuggingFace xLSTM design).
    """

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            use_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._eps = mx.array(eps, dtype=mx.float32)
        self.use_weight = use_weight
        self.use_bias = use_bias

        # CRITICAL: Weight and bias are FLAT [num_heads * head_dim], not [num_heads, head_dim]!
        # This matches PyTorch transformers xLSTMMultiHeadLayerNorm
        if use_weight:
            self.weight = mx.ones((num_heads * head_dim))
        if use_bias:
            self.bias = mx.zeros((num_heads * head_dim))

    def __call__(self, x: mx.array) -> mx.array:  # noqa: D401
        """Normalize per head, flatten, and scale.

        Parameters
        ----------
        x : mx.array [B, S, NH, DH]
            Multi-head input tensor.

        Returns
        -------
        output : mx.array [B, S, NH * DH]
            Normalized and flattened output.
        """
        B, S, NH, DH = x.shape
        if NH != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} heads, got {NH}")
        if DH != self.head_dim:
            raise ValueError(f"Expected {self.head_dim} head_dim, got {DH}")

        input_dtype = x.dtype

        # Force float32 for reductions if requested
        if self.force_float32_reductions:
            x = x.astype(mx.float32)

        # Normalize per head: compute mean/var over head_dim dimension (last dim)
        mean = mx.mean(x, axis=-1, keepdims=True)  # [B, S, NH, 1]
        variance = mx.var(x, axis=-1, keepdims=True)  # [B, S, NH, 1]

        eps_t = self._eps.astype(x.dtype)
        x_norm = mx.multiply(mx.subtract(x, mean), mx.rsqrt(mx.add(variance, eps_t)))

        # Cast back to input dtype
        x_norm = x_norm.astype(input_dtype)

        # CRITICAL: Reshape BEFORE applying weight/bias (matches transformers!)
        # PyTorch: normalize per-head, reshape to flat, THEN apply weight
        x_norm = x_norm.reshape(B, S, -1)  # [B, S, NH*DH]

        # Apply weight and bias to FLAT tensor
        # weight: [NH*DH], x_norm: [B, S, NH*DH]
        if self.use_weight:
            x_norm = mx.multiply(self.weight, x_norm)
        if self.use_bias:
            x_norm = mx.add(self.bias, x_norm)

        return x_norm


class MultiHeadRMSNorm(nn.Module):
    """Per-head RMSNorm with flattening (RMS scaling only, no mean centering).

    Applies RMS normalization independently to each head's features, then
    flattens and applies a shared weight vector. Preferred for efficiency
    when mean centering is not required.

    Per-Head RMS Computation
    ------------------------
    For each head h:
      RMS_h = sqrt(mean(x[:,:,h,:]²) + eps)
      x_norm[:,:,h,:] = x[:,:,h,:] / RMS_h

    Parameters
    ----------
    num_heads : int
        Number of attention heads (NH).
    head_dim : int
        Dimension per head (DH).
    eps : float, default 1e-6
        Numerical stability epsilon for RMS.
    use_weight : bool, default True
        Whether to apply learnable weight scaling.
    force_float32_reductions : bool, default True
        Force float32 accumulation in RMS computation (recommended).

    Returns (forward)
    -----------------
    output : mx.array [B, S, NH * DH]
        RMS-normalized and flattened activations.

    Notes
    -----
    Weight is flat [NH * DH], applied **after** per-head RMS normalization
    and flattening. This is the standard pattern in xLSTM output cells.
    """

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            force_float32_reductions: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._eps = mx.array(eps, dtype=mx.float32)
        self.use_weight = use_weight
        self.force_float32_reductions = force_float32_reductions

        # Weight is flat [num_heads * head_dim]
        if use_weight:
            self.weight = mx.ones((num_heads * head_dim,))

    def __call__(self, x: mx.array) -> mx.array:  # noqa: D401
        """RMS normalize per head, flatten, and scale.

        Parameters
        ----------
        x : mx.array [B, S, NH, DH]
            Multi-head input tensor.

        Returns
        -------
        output : mx.array [B, S, NH * DH]
            RMS-normalized and flattened output.
        """
        B, S, NH, DH = x.shape
        if NH != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} heads, got {NH}")
        if DH != self.head_dim:
            raise ValueError(f"Expected {self.head_dim} head_dim, got {DH}")

        input_dtype = x.dtype

        # Force float32 for reductions if requested
        if self.force_float32_reductions:
            x = x.astype(mx.float32)

        # RMS normalize per head: compute RMS over head_dim (last dim)
        # RMS = sqrt(mean(x^2))
        rms = mx.sqrt(mx.add(mx.mean(mx.square(x), axis=-1, keepdims=True), self._eps.astype(x.dtype)))
        x_norm = mx.divide(x, rms)

        # Cast back to input dtype
        x_norm = x_norm.astype(input_dtype)

        # Reshape to flat before applying weight
        x_norm = x_norm.reshape(B, S, -1)  # [B, S, NH*DH]

        # Apply weight to flat tensor
        if self.use_weight:
            x_norm = mx.multiply(self.weight, x_norm)

        return x_norm


__all__ = ['MultiHeadLayerNorm', 'MultiHeadRMSNorm']
