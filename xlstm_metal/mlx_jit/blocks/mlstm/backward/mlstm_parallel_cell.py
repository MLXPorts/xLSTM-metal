"""MLX implementation of mLSTM Parallel Cell for NCPS.

This cell implements matrix LSTM using parallel (chunkwise) operations,
compatible with the NCPS wiring framework.

This is a true NCPS "cell" that processes sequences using parallel kernels.
All operations use Metal acceleration via MLX.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Use dynamic kernel registry instead of static import
from xlstm_metal.blocks.mlstm.kern import get_kernel


class mLSTMParallelCell(nn.Module):
    """
    Matrix LSTM Parallel Cell for NCPS.
    
    This cell uses parallel (chunkwise) operations to process sequences efficiently.
    It's designed as a true NCPS "cell" that can be wrapped by higher-level modules.
    
    Components:
    - Q/K/V projections (feature groups)
    - Input/forget/output gates (gating neurons)
    - Parallel memory kernel (processes chunks in parallel)
    - Output projection
    
    Args:
        input_size: Input dimension (embedding_dim)
        hidden_size: Hidden dimension per head
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head
        v_dim_per_head: Value dimension per head
        chunk_size: Chunk size for parallel kernel
        kernel_type: Kernel type ('metal_jit', 'mlx_native', 'chunkwise')
        use_bias: Whether to use bias in projections
        eps: Numerical stability epsilon
        sparsity_mask: Optional sparsity mask for NCPS wiring
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_heads: int = 8,
            qk_dim_per_head: Optional[int] = None,
            v_dim_per_head: Optional[int] = None,
            chunk_size: int = 64,
            kernel_type: str = "metal_jit",
            use_bias: bool = False,
            eps: float = 1e-6,
            sparsity_mask: Optional[mx.array] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.kernel_type = kernel_type
        self.eps = eps

        # Resolve kernel at init time (fail fast with clear error)
        # Maps kernel_type to (algorithm, backend) for registry lookup
        kernel_map = {
            "metal_jit": ("chunkwise", "metal_jit"),
            "mlx_native": ("chunkwise", "mlx_native"),
            "chunkwise": ("chunkwise", "metal_jit"),  # Default to metal_jit
        }
        
        if kernel_type not in kernel_map:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}', "
                f"valid options are {set(kernel_map.keys())}"
            )
        
        algorithm, backend = kernel_map[kernel_type]
        try:
            self._kernel_fn = get_kernel(algorithm, backend)
        except (ValueError, NotImplementedError) as e:
            raise ValueError(
                f"Failed to resolve kernel for type '{kernel_type}': {e}"
            ) from e

        # Validate kernel type (kept for backward compatibility)
        allowed_kernel_types = {"metal_jit", "mlx_native", "chunkwise"}
        if kernel_type not in allowed_kernel_types:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}', "
                f"valid options are {allowed_kernel_types}"
            )

        # Compute dimensions
        self.qk_dim_per_head = qk_dim_per_head or (hidden_size // num_heads)
        self.v_dim_per_head = v_dim_per_head or hidden_size

        self.qk_dim = num_heads * self.qk_dim_per_head
        self.v_dim = num_heads * self.v_dim_per_head

        # Q/K/V projections (feature groups in NCPS terminology)
        self.q_proj = nn.Linear(input_size, self.qk_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_size, self.qk_dim, bias=use_bias)
        self.v_proj = nn.Linear(input_size, self.v_dim, bias=use_bias)

        # Input/forget gates (excitatory/inhibitory neurons)
        self.igate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(input_size, num_heads, bias=True)

        # Output gate (modulates output)
        self.ogate_proj = nn.Linear(input_size, self.v_dim, bias=use_bias)

        # Multi-head layer norm
        self.norm = nn.RMSNorm(self.v_dim_per_head, eps=eps)

        # Output projection
        self.out_proj = nn.Linear(self.v_dim, input_size, bias=use_bias)

        # Sparsity mask
        if sparsity_mask is not None:
            mask = mx.array(sparsity_mask) if not isinstance(sparsity_mask, mx.array) else sparsity_mask
            self._sparsity_mask = mx.abs(mask).astype(mx.float32)
        else:
            self._sparsity_mask = None

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array, mx.array]]:
        """
        Forward pass through mLSTM parallel cell.
        
        Args:
            x: Input [B, S, input_size]
            state: Optional previous state (C, n, m) for recurrence
            
        Returns:
            (output, new_state): Output [B, S, input_size] and new state (C, n, m)
        """
        B, S, _ = x.shape

        # Project to Q/K/V (feature groups)
        q = self.q_proj(x)  # [B, S, qk_dim]
        k = self.k_proj(x)  # [B, S, qk_dim]
        v = self.v_proj(x)  # [B, S, v_dim]

        # Reshape to multi-head
        q = q.reshape(B, S, self.num_heads, self.qk_dim_per_head)
        k = k.reshape(B, S, self.num_heads, self.qk_dim_per_head)
        v = v.reshape(B, S, self.num_heads, self.v_dim_per_head)

        # Gates (inhibitory fgate with polarity -1 in NCPS)
        i_gate = mx.exp(self.igate_proj(x))  # [B, S, num_heads] - excitatory
        f_gate = mx.exp(self.fgate_proj(x))  # [B, S, num_heads] - inhibitory

        # Initialize state if needed
        if state is None:
            C = mx.zeros((B, self.num_heads, self.qk_dim_per_head, self.v_dim_per_head))
            n = mx.zeros((B, self.num_heads, self.qk_dim_per_head))
            m = mx.zeros((B, self.num_heads))
        else:
            C, n, m = state

        # Apply parallel kernel (configured kernel type)
        h, C_new, n_new, m_new = self._apply_parallel_kernel(
            q, k, v, i_gate, f_gate, C, n, m
        )

        # Apply sparsity mask if present
        if self._sparsity_mask is not None:
            h = mx.multiply(h, self._sparsity_mask)

        # Normalize (per-head)
        h = h.reshape(B, S, self.num_heads, self.v_dim_per_head)
        h_norm = self.norm(h)

        # Output gate modulation
        o_gate = mx.sigmoid(self.ogate_proj(x))  # [B, S, v_dim]
        o_gate = o_gate.reshape(B, S, self.num_heads, self.v_dim_per_head)
        h_gated = h_norm * o_gate

        # Flatten heads and project to output
        h_flat = h_gated.reshape(B, S, self.v_dim)
        output = self.out_proj(h_flat)  # [B, S, input_size]

        # Return output and new state
        new_state = (C_new, n_new, m_new)
        return output, new_state

    def _apply_parallel_kernel(
            self,
            q: mx.array,
            k: mx.array,
            v: mx.array,
            i_gate: mx.array,
            f_gate: mx.array,
            C: mx.array,
            n: mx.array,
            m: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Apply parallel (chunkwise) kernel for mLSTM.
        
        Uses the kernel resolved at init time via the registry.
        """
        # Call the resolved kernel function directly
        return self._kernel_fn(q, k, v, i_gate, f_gate, C, n, m)

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "qk_dim_per_head": self.qk_dim_per_head,
            "v_dim_per_head": self.v_dim_per_head,
            "chunk_size": self.chunk_size,
            "kernel_type": self.kernel_type,
            "eps": self.eps,
        }
