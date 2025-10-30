"""
mLSTM Block for PyTorch Backend

Implements mLSTM layer using native PyTorch kernels from kernel_development.
Matches xLSTM-7B weight structure for compatibility with MLX version.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn


@dataclass
class mLSTMConfig:
    """
    Configuration for PyTorch mLSTM block.
    Matches MLX version for weight compatibility.
    """
    embedding_dim: int = 4096
    num_heads: int = 8
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    use_bias: bool = False
    norm_eps: float = 1e-6
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True
    chunk_size: int = 64

    def __post_init__(self):
        self.qk_dim = int(self.embedding_dim * self.qk_dim_factor)
        self.v_dim = int(self.embedding_dim * self.v_dim_factor)
        self.head_dim = self.v_dim // self.num_heads
        self.qk_head_dim = self.qk_dim // self.num_heads


class LayerNorm(nn.Module):
    """Per-head layer norm (like MultiHeadLayerNorm in MLX version)."""
    
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, NH, S, head_dim]
        Returns:
            [B, NH, S, head_dim]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


def soft_cap(x: torch.Tensor, soft_cap_value: float) -> torch.Tensor:
    """Soft capping: tanh(x / soft_cap) * soft_cap."""
    return torch.tanh(x / soft_cap_value) * soft_cap_value


class mLSTMLayer(nn.Module):
    """
    PyTorch mLSTM Layer.
    
    Uses native PyTorch kernels from kernel_development/mlstm_block/mlstm_recurrent.
    Matches xLSTM-7B weight structure for checkpoint compatibility.
    """

    def __init__(self, config: mLSTMConfig):
        super().__init__()
        self.config = config

        # QKV projections
        self.q = nn.Linear(config.embedding_dim, config.qk_dim, bias=config.use_bias)
        self.k = nn.Linear(config.embedding_dim, config.qk_dim, bias=config.use_bias)
        self.v = nn.Linear(config.embedding_dim, config.v_dim, bias=config.use_bias)

        # Input and forget gates (per-head)
        self.igate_preact = nn.Linear(config.embedding_dim, config.num_heads)
        self.fgate_preact = nn.Linear(config.embedding_dim, config.num_heads)

        # Output gate
        self.ogate_preact = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.use_bias)

        # Normalization (per-head)
        self.multihead_norm = LayerNorm(config.num_heads, config.head_dim, eps=config.norm_eps)

        # Output projection
        self.out_proj = nn.Linear(config.v_dim, config.embedding_dim, bias=config.use_bias)

        # Import native kernel
        try:
            from kernel_development.mlstm_block.mlstm_recurrent.recurrent.native_sequence import (
                mlstm_recurrent_sequence_loop
            )
            self.mlstm_kernel = mlstm_recurrent_sequence_loop
        except ImportError:
            # Fallback to importing from path
            import sys
            from pathlib import Path
            kernel_path = Path(__file__).parent.parent.parent.parent / "kernel_development"
            sys.path.insert(0, str(kernel_path))
            try:
                from mlstm_block.mlstm_recurrent.recurrent.native_sequence import (
                    mlstm_recurrent_sequence_loop
                )
                self.mlstm_kernel = mlstm_recurrent_sequence_loop
            except ImportError:
                print("Warning: Could not import native mLSTM kernel. Using pure Python fallback.")
                self.mlstm_kernel = None

    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass through mLSTM layer.
        
        Args:
            x: Input tensor [B, S, D]
            state: Optional initial state (C, N, M)
        
        Returns:
            output: [B, S, D]
            new_state: (C, N, M) for next step
        """
        B, S, D = x.shape

        # Project QKV
        q = self.q(x)  # [B, S, qk_dim]
        k = self.k(x)  # [B, S, qk_dim]
        v = self.v(x)  # [B, S, v_dim]

        # Reshape to multi-head
        q = q.reshape(B, S, self.config.num_heads, self.config.qk_head_dim).transpose(1, 2)  # [B, NH, S, qk_dim]
        k = k.reshape(B, S, self.config.num_heads, self.config.qk_head_dim).transpose(1, 2)  # [B, NH, S, qk_dim]
        v = v.reshape(B, S, self.config.num_heads, self.config.head_dim).transpose(1, 2)  # [B, NH, S, head_dim]

        # Gates
        i_preact = self.igate_preact(x)  # [B, S, NH]
        f_preact = self.fgate_preact(x)  # [B, S, NH]
        
        # Reshape gates to [B, NH, S]
        i = i_preact.transpose(1, 2)  # [B, NH, S]
        f = f_preact.transpose(1, 2)  # [B, NH, S]

        # Get initial states if provided
        if state is not None:
            c_init, n_init, m_init = state
        else:
            c_init, n_init, m_init = None, None, None

        # Call mLSTM kernel
        if self.mlstm_kernel is not None:
            h, n, m, _, _ = self.mlstm_kernel(
                q, k, v, i, f,
                matC_initial=c_init,
                vecN_initial=n_init,
                scaM_initial=m_init,
                return_last_states=True,
                return_all_states=False,
                eps=self.config.eps,
                dtype_state=torch.float32
            )
        else:
            # Fallback: use simpler recurrent implementation
            h = self._recurrent_fallback(q, k, v, i, f, c_init, n_init, m_init)
            n, m = None, None

        # Normalize heads
        h_norm = self.multihead_norm(h)  # [B, NH, S, head_dim]

        # Reshape back to sequence
        h_norm = h_norm.transpose(1, 2).contiguous()  # [B, S, NH, head_dim]
        h_norm = h_norm.reshape(B, S, self.config.v_dim)  # [B, S, v_dim]

        # Output gate
        o_preact = self.ogate_preact(x)  # [B, S, D]
        o = torch.sigmoid(soft_cap(o_preact, self.config.gate_soft_cap))  # [B, S, D]

        # Output projection
        out = self.out_proj(h_norm * o)  # [B, S, D]

        # Prepare new state
        new_state = (h, n, m) if (n is not None and m is not None) else state

        return out, new_state

    def _recurrent_fallback(
        self,
        q: torch.Tensor,  # [B, NH, S, qk_dim]
        k: torch.Tensor,  # [B, NH, S, qk_dim]
        v: torch.Tensor,  # [B, NH, S, head_dim]
        i: torch.Tensor,  # [B, NH, S]
        f: torch.Tensor,  # [B, NH, S]
        c_init: Optional[torch.Tensor] = None,
        n_init: Optional[torch.Tensor] = None,
        m_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fallback recurrent implementation if kernel not available.
        
        Args:
            q, k, v, i, f: Reshaped inputs
            c_init, n_init, m_init: Initial states
        
        Returns:
            h: [B, NH, S, head_dim]
        """
        B, NH, S, qk_dim = q.shape
        head_dim = v.shape[-1]
        device = q.device

        # Initialize states
        if c_init is None:
            c_state = torch.zeros(B, NH, qk_dim, head_dim, dtype=torch.float32, device=device)
            n_state = torch.zeros(B, NH, qk_dim, dtype=torch.float32, device=device)
            m_state = torch.zeros(B, NH, 1, dtype=torch.float32, device=device)
        else:
            c_state = c_init.to(dtype=torch.float32)
            n_state = n_init.to(dtype=torch.float32)
            m_state = m_init.to(dtype=torch.float32)

        # Recurrent loop
        h_list = []
        for t in range(S):
            q_t = q[:, :, t, :]  # [B, NH, qk_dim]
            k_t = k[:, :, t, :]  # [B, NH, qk_dim]
            v_t = v[:, :, t, :]  # [B, NH, head_dim]
            i_t = i[:, :, t:t+1]  # [B, NH, 1]
            f_t = f[:, :, t:t+1]  # [B, NH, 1]

            # Update logic (simplified exponential gating)
            f_log = torch.nn.functional.logsigmoid(f_t)
            m_new = torch.max(f_log + m_state, i_t)
            
            f_act = torch.exp(f_log + m_state - m_new)
            i_act = torch.exp(i_t - m_new)

            # Update C state
            kv = torch.einsum('...i,... j->...ij', k_t, v_t)  # [B, NH, qk_dim, head_dim]
            c_new = f_act[:, :, :, None] * c_state + i_act[:, :, :, None] * kv

            # Output
            q_scaled = q_t * (qk_dim ** -0.5)
            h_t = torch.einsum('...ij,...i->...j', c_new, q_scaled)  # [B, NH, head_dim]

            h_list.append(h_t.unsqueeze(2))
            c_state, m_state = c_new, m_new

        h = torch.cat(h_list, dim=2)  # [B, NH, S, head_dim]
        return h
