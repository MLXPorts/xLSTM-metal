import mlx.core as mx
import mlx.nn as nn

class MultiHeadLayerNorm(nn.Module):
    """
    Multi-Head Layer Normalization

    Normalizes independently per head, not across all dimensions.
    Critical for xLSTM-7B mLSTM blocks.

    Weight shape: [num_heads, head_dim] not [num_heads * head_dim]!
    """

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            use_bias: bool = False,
            force_float32_reductions: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._eps = mx.array(eps, dtype=mx.float32)
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.force_float32_reductions = force_float32_reductions

        # CRITICAL: Weight and bias are FLAT [num_heads * head_dim], not [num_heads, head_dim]!
        # This matches PyTorch transformers xLSTMMultiHeadLayerNorm
        if use_weight:
            self.weight = mx.ones((num_heads * head_dim,))
        if use_bias:
            self.bias = mx.zeros((num_heads * head_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass matching transformers xLSTMMultiHeadLayerNorm

        Args:
            x: Input tensor [B, S, num_heads, head_dim]

        Returns:
            Normalized tensor [B, S, num_heads * head_dim]
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
