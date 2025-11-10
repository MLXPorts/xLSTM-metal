"""MLX implementation of mLSTM as an NCPS cell with Metal kernel support.

This cell implements matrix LSTM with optional Metal acceleration,
compatible with the NCPS wiring framework.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.blocks.mlstm_layer.kernels import (
    mlstm_chunkwise_parallel_fw_Hintra_metal,
    mlstm_chunkwise_recurrent_fw_C_n_m_metal,
)


class mLSTMCell(nn.Module):
    """
    Matrix LSTM cell for NCPS with Metal kernel support.
    
    This is a composite cell that contains multiple sub-components:
    - Q/K/V projections (feature groups)
    - Input/forget/output gates (gating neurons)
    - Memory kernel (recurrent core)
    - Output projection
    
    The cell can use Metal-accelerated kernels when available.
    
    Args:
        input_size: Input dimension (embedding_dim)
        hidden_size: Hidden dimension per head
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head
        v_dim_per_head: Value dimension per head
        kernel_backend: Kernel backend ('metal_parallel', 'metal_sequential', 'pure_mlx')
        chunk_size: Chunk size for parallel kernel
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
            kernel_backend: str = "metal_parallel",
            chunk_size: int = 64,
            use_bias: bool = False,
            eps: float = 1e-6,
            sparsity_mask: Optional[mx.array] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_backend = kernel_backend
        self.chunk_size = chunk_size
        self.eps = eps

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

        # Check kernel availability
        if kernel_backend.startswith("metal") and not METAL_KERNELS_AVAILABLE:
            print(f"Warning: Metal kernels not available, falling back to pure_mlx")
            self.kernel_backend = "pure_mlx"

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
        Forward pass through mLSTM cell.
        
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

        # Apply kernel (memory update - the recurrent core)
        if self.kernel_backend == "metal_parallel":
            h, C_new, n_new, m_new = self._apply_metal_parallel_kernel(
                q, k, v, i_gate, f_gate, C, n, m
            )
        elif self.kernel_backend == "metal_sequential":
            h, C_new, n_new, m_new = self._apply_metal_sequential_kernel(
                q, k, v, i_gate, f_gate, C, n, m
            )
        elif self.kernel_backend == "pure_mlx":
            h, C_new, n_new, m_new = self._apply_pure_mlx_kernel(
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

    def _apply_metal_parallel_kernel(
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
        """Apply Metal parallel kernel."""
        # This would call the compiled Metal kernel
        # For now, placeholder that falls back to pure MLX
        return self._apply_pure_mlx_kernel(q, k, v, i_gate, f_gate, C, n, m)

    def _apply_metal_sequential_kernel(
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
        """Apply Metal sequential kernel."""
        # This would call the compiled Metal kernel
        return self._apply_pure_mlx_kernel(q, k, v, i_gate, f_gate, C, n, m)

    def _apply_pure_mlx_kernel(
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
        """Pure MLX implementation (fallback)."""
        B, S, H, D_qk = q.shape
        _, _, _, D_v = v.shape

        # Simple linear attention as fallback
        # h_t = C_t @ q_t where C_t = f_t * C_{t-1} + i_t * (k_t @ v_t^T)

        C_new = C
        n_new = n
        m_new = m

        h_list = []
        for t in range(S):
            q_t = q[:, t, :, :]  # [B, H, D_qk]
            k_t = k[:, t, :, :]  # [B, H, D_qk]
            v_t = v[:, t, :, :]  # [B, H, D_v]
            i_t = i_gate[:, t, :].reshape(B, H, 1, 1)  # [B, H, 1, 1]
            f_t = f_gate[:, t, :].reshape(B, H, 1, 1)  # [B, H, 1, 1]

            # Update: C_t = f_t * C_{t-1} + i_t * (k_t @ v_t^T)
            kv = k_t[:, :, :, None] * v_t[:, :, None, :]  # [B, H, D_qk, D_v]
            C_new = f_t * C_new + i_t * kv

            # Compute output: h_t = C_t @ q_t
            h_t = mx.sum(C_new * q_t[:, :, :, None], axis=2)  # [B, H, D_v]
            h_list.append(h_t)

        h = mx.stack(h_list, axis=1)  # [B, S, H, D_v]

        return h, C_new, n_new, m_new

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "qk_dim_per_head": self.qk_dim_per_head,
            "v_dim_per_head": self.v_dim_per_head,
            "kernel_backend": self.kernel_backend,
            "chunk_size": self.chunk_size,
            "eps": self.eps,
        }
