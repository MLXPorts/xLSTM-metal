#!/usr/bin/env python
"""
mLSTM Cell - Core Computational Unit

Following NCPS architecture:
- Cell handles the core mLSTM computation
- Can accept wiring for sparsity patterns
- Parameters automatically tracked by nn.Module
- Block wrapper adds norms and residuals
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from .components import MultiHeadLayerNorm, soft_cap
from .kernel import mlstm_chunkwise


@dataclass
class mLSTMCellConfig:
    """Configuration for mLSTM cell."""
    embedding_dim: int
    num_heads: int
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True
    chunk_size: int = 64

    def __post_init__(self):
        # Compute derived dimensions
        self.qk_dim = int(self.embedding_dim * self.qk_dim_factor)
        self.v_dim = int(self.embedding_dim * self.v_dim_factor)
        self.head_dim = self.qk_dim // self.num_heads

        assert self.qk_dim % self.num_heads == 0, \
            f"qk_dim ({self.qk_dim}) must be divisible by num_heads ({self.num_heads})"


class mLSTMCell(nn.Module):
    """
    mLSTM Cell - Matrix Memory LSTM Core.

    This is the actual computational unit, following NCPS cell pattern.
    The cell handles:
    - Query/Key/Value projections
    - Input/Forget/Output gates
    - Matrix memory updates
    - Multi-head layer normalization

    Can accept optional wiring for:
    - Sparse projections
    - Weight sharing patterns
    - Custom connectivity

    Based on:
    - xLSTM paper equations
    - NCPS cell architecture
    - LTC/CfC cell patterns
    """

    def __init__(
        self,
        config: mLSTMCellConfig,
        wiring: Optional[Any] = None,
        sparsity_mask: Optional[mx.array] = None
    ):
        """
        Initialize mLSTM cell.

        Args:
            config: Cell configuration
            wiring: Optional wiring for connectivity patterns
            sparsity_mask: Optional sparsity mask for projections
        """
        super().__init__()

        self.config = config
        self._wiring = wiring
        self._sparsity_mask = sparsity_mask

        # Query, Key, Value projections
        # These can be masked by sparsity patterns from wiring
        self.q = nn.Linear(
            config.embedding_dim,
            config.qk_dim,
            bias=config.use_bias
        )
        self.k = nn.Linear(
            config.embedding_dim,
            config.qk_dim,
            bias=config.use_bias
        )
        self.v = nn.Linear(
            config.embedding_dim,
            config.v_dim,
            bias=config.use_bias
        )

        # Gates: input, forget, output
        # Using separate weight and bias for each gate head
        self.igate_preact = nn.Linear(
            config.embedding_dim,
            config.num_heads,
            bias=True  # Gates need bias
        )
        self.fgate_preact = nn.Linear(
            config.embedding_dim,
            config.num_heads,
            bias=True
        )
        self.ogate_preact = nn.Linear(
            config.embedding_dim,
            config.num_heads,
            bias=True  # Added for completeness
        )

        # Multi-head layer normalization
        # Normalizes across head dimension
        self.multihead_norm = MultiHeadLayerNorm(
            num_features=config.num_heads * config.head_dim,
            num_heads=config.num_heads,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=False,
            force_float32_reductions=config.norm_reduction_force_float32
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.v_dim,
            config.embedding_dim,
            bias=config.use_bias
        )

    def _apply_sparsity(self, weight: mx.array) -> mx.array:
        """Apply sparsity mask if provided."""
        if self._sparsity_mask is not None:
            return mx.multiply(weight, self._sparsity_mask)
        return weight

    @property
    def state_size(self) -> Tuple[int, int, int]:
        """
        Return mLSTM state dimensions.

        State consists of:
        - C: [num_heads, head_dim, head_dim] - covariance matrix
        - n: [num_heads, head_dim] - normalizer
        - m: [num_heads, 1] - running max
        """
        return (
            (self.config.num_heads, self.config.head_dim, self.config.head_dim),
            (self.config.num_heads, self.config.head_dim),
            (self.config.num_heads, 1)
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass through mLSTM cell.

        Args:
            x: Input [B, S, embedding_dim]
            state: Optional (C, n, m) state tuple

        Returns:
            output: [B, S, embedding_dim]
            new_state: Updated (C, n, m) or None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        # Apply sparsity if wiring provides it
        q = self.q(x)  # [B, S, qk_dim]
        k = self.k(x)  # [B, S, qk_dim]
        v = self.v(x)  # [B, S, v_dim]

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        k = k.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        v = v.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        # Compute gates
        igate = self.igate_preact(x)  # [B, S, num_heads]
        fgate = self.fgate_preact(x)  # [B, S, num_heads]

        # Apply soft cap to gate pre-activations
        igate = soft_cap(igate, self.config.gate_soft_cap)
        fgate = soft_cap(fgate, self.config.gate_soft_cap)

        # Apply sigmoid to get gate values
        igate = mx.sigmoid(igate)
        fgate = mx.sigmoid(fgate)

        # Compute stabilized forget gate (exponential space)
        # f_tilde = exp(log(f))  where log(f) is stabilized
        log_fgate = mx.log(mx.maximum(fgate, self.config.eps))

        # Run mLSTM kernel (chunkwise processing)
        h, new_state = mlstm_chunkwise(
            q=q,
            k=k,
            v=v,
            igate=igate,
            fgate=log_fgate,
            state=state,
            chunk_size=self.config.chunk_size,
            return_last_states=self.config.return_last_states
        )

        # h shape: [B, S, num_heads, head_dim]

        # Flatten heads for layer norm
        h_flat = h.reshape(batch_size, seq_len, self.config.num_heads * self.config.head_dim)

        # Multi-head layer normalization
        h_norm = self.multihead_norm(h_flat)

        # Output gate
        ogate = self.ogate_preact(x)  # [B, S, num_heads]
        ogate = mx.sigmoid(ogate)

        # Apply output gate per head
        # Reshape ogate to match h_norm structure
        ogate_expanded = ogate.reshape(batch_size, seq_len, self.config.num_heads, 1)
        ogate_expanded = mx.tile(ogate_expanded, [1, 1, 1, self.config.head_dim])
        ogate_flat = ogate_expanded.reshape(batch_size, seq_len, self.config.num_heads * self.config.head_dim)

        h_gated = h_norm * ogate_flat

        # Output projection
        output = self.out_proj(h_gated)

        return output, new_state

    def reset_state(self, batch_size: int = 1) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Initialize mLSTM state.

        Args:
            batch_size: Batch size

        Returns:
            Initial (C, n, m) state
        """
        C = mx.zeros((batch_size, self.config.num_heads, self.config.head_dim, self.config.head_dim))
        n = mx.zeros((batch_size, self.config.num_heads, self.config.head_dim))
        m = mx.full((batch_size, self.config.num_heads, 1), -float('inf'))

        return (C, n, m)

