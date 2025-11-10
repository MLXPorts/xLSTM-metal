"""MLX implementation of mLSTM as an NCPS cell with Metal kernel support.

This cell implements matrix LSTM with optional Metal acceleration,
compatible with the NCPS wiring framework.

Critical implementation details (matches xLSTM-7B transformers chunkwise):
- Forget gate uses logsigmoid before exponential
- Key is NOT scaled when storing in C and n (xLSTM-7B approach)
- Query IS scaled by 1/√d_qk during retrieval (xLSTM-7B approach)
- Denominator uses max(|q_scaled·n|, exp(-m)) + eps
- C state shape is [B, NH, DHQK, DHHV] (k⊗v not v⊗k)
- States maintained in float32 for numerical stability
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Import kernel functions
from xlstm_metal.mlx_jit.blocks.mlstm.kernel import (
    mlstm_sequential,
    mlstm_chunkwise,
)


class mLSTMCell(nn.Module):
    """
    Matrix LSTM cell for NCPS with Metal kernel support.
    
    This is a composite cell that contains multiple sub-components:
    - Q/K/V projections (feature groups)
    - Input/forget/output gates (gating neurons)
    - Memory kernel (recurrent core using proven kernel.py patterns)
    - Output projection
    
    The cell uses production kernel implementations from kernel.py:
    - metal_chunkwise: Two-phase chunkwise parallel (8-55x speedup on M3 Ultra)
    - metal_sequential: Step-by-step recurrence (for inference)
    - pure_mlx: Pure MLX fallback (for portability)
    
    Args:
        input_size: Input dimension (embedding_dim)
        hidden_size: Hidden dimension per head
        num_heads: Number of attention heads
        qk_dim_per_head: Query/key dimension per head (default: hidden_size // num_heads)
        v_dim_per_head: Value dimension per head (default: hidden_size)
        kernel_mode: Kernel mode ('metal_chunkwise', 'metal_sequential', 'pure_mlx')
        chunk_size: Chunk size for chunkwise parallel kernel (default: 64)
        use_bias: Whether to use bias in projections (default: False)
        eps: Numerical stability epsilon (default: 1e-6)
        sparsity_mask: Optional sparsity mask for NCPS wiring
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_heads: int = 8,
            qk_dim_per_head: Optional[int] = None,
            v_dim_per_head: Optional[int] = None,
            kernel_mode: str = "metal_chunkwise",
            chunk_size: int = 64,
            use_bias: bool = False,
            eps: float = 1e-6,
            sparsity_mask: Optional[mx.array] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_mode = kernel_mode
        self.chunk_size = chunk_size
        self.eps = eps

        # Validate kernel mode
        valid_modes = {"metal_chunkwise", "metal_sequential", "pure_mlx"}
        if kernel_mode not in valid_modes:
            raise ValueError(
                f"Invalid kernel_mode '{kernel_mode}'. "
                f"Valid options: {sorted(valid_modes)}"
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
        Forward pass through mLSTM cell.
        
        Tensor shapes follow kernel.py canonical format:
        - Q/K/V: [B, NH, S, DH] (batch, heads, sequence, dim_per_head)
        - Gates: [B, NH, S] (pre-activation, no exp yet)
        - C state: [B, NH, DHQK, DHHV] (k⊗v format, NOT v⊗k)
        - n state: [B, NH, DHQK]
        - m state: [B, NH]
        
        Args:
            x: Input [B, S, input_size]
            state: Optional previous state (C, n, m) for recurrence
                   C: [B, NH, DHQK, DHHV]
                   n: [B, NH, DHQK]
                   m: [B, NH]
            
        Returns:
            (output, new_state): Output [B, S, input_size] and new state (C, n, m)
        """
        B, S, _ = x.shape

        # Project to Q/K/V (feature groups)
        q = self.q_proj(x)  # [B, S, qk_dim]
        k = self.k_proj(x)  # [B, S, qk_dim]
        v = self.v_proj(x)  # [B, S, v_dim]

        # Reshape to multi-head: [B, S, NH, DH] -> [B, NH, S, DH]
        q = q.reshape(B, S, self.num_heads, self.qk_dim_per_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.qk_dim_per_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.v_dim_per_head).transpose(0, 2, 1, 3)

        # Gates: [B, S, NH] -> [B, NH, S]
        # CRITICAL: Pass pre-activations to kernel (no exp here!)
        # kernel.py applies logsigmoid(f) and then exp for numerical stability
        i_preact = self.igate_proj(x).transpose(0, 2, 1)  # [B, NH, S]
        f_preact = self.fgate_proj(x).transpose(0, 2, 1)  # [B, NH, S]

        # Initialize state if needed
        # CRITICAL: States must be float32 for numerical stability (matches kernel.py)
        if state is None:
            c_initial = None
            n_initial = None
            m_initial = None
        else:
            c_initial, n_initial, m_initial = state
            # Ensure float32 (kernel.py requirement)
            c_initial = c_initial.astype(mx.float32) if c_initial is not None else None
            n_initial = n_initial.astype(mx.float32) if n_initial is not None else None
            m_initial = m_initial.astype(mx.float32) if m_initial is not None else None

        # Apply kernel (memory update - the recurrent core)
        # Dispatch to kernel.py implementations
        if self.kernel_mode == "metal_chunkwise":
            # Use two-phase chunkwise parallel kernel (8-55x speedup on M3 Ultra)
            h, final_state = mlstm_chunkwise(
                q=q, k=k, v=v,
                i_preact=i_preact,
                f_preact=f_preact,
                chunk_size=self.chunk_size,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
                eps=self.eps,
                return_last_states=True,
            )
            C_new, n_new, m_new = final_state
        elif self.kernel_mode == "metal_sequential":
            # Use step-by-step sequential kernel
            h, final_state = mlstm_sequential(
                q=q, k=k, v=v,
                i_preact=i_preact,
                f_preact=f_preact,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
                eps=self.eps,
                return_last_states=True,
            )
            C_new, n_new, m_new = final_state
        elif self.kernel_mode == "pure_mlx":
            # Use pure MLX fallback (same as metal_sequential but explicit)
            h, final_state = mlstm_sequential(
                q=q, k=k, v=v,
                i_preact=i_preact,
                f_preact=f_preact,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
                eps=self.eps,
                return_last_states=True,
            )
            C_new, n_new, m_new = final_state
        else:
            raise ValueError(f"Unknown kernel_mode: {self.kernel_mode}")

        # h is now [B, NH, S, V_DH], need to transpose back to [B, S, NH, V_DH]
        h = h.transpose(0, 2, 1, 3)  # [B, S, NH, V_DH]

        # Apply sparsity mask if present
        if self._sparsity_mask is not None:
            h = mx.multiply(h, self._sparsity_mask)

        # Normalize (per-head)
        h_norm = self.norm(h)  # [B, S, NH, V_DH]

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

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "qk_dim_per_head": self.qk_dim_per_head,
            "v_dim_per_head": self.v_dim_per_head,
            "kernel_mode": self.kernel_mode,
            "chunk_size": self.chunk_size,
            "eps": self.eps,
        }

