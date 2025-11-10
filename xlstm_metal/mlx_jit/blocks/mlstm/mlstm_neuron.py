"""mLSTM neuron for NCPS - simple recurrent unit.

This is a basic mLSTM neuron that can be wired together using NCPS patterns.
Direct Metal kernel integration - no fallbacks, no abstraction layers.
"""

from __future__ import annotations
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

# Direct Metal kernel imports
from xlstm_metal.mlx_jit.blocks.mlstm.forward.mlstm_chunkwise_recurrent_fw_C import (
    mlstm_chunkwise_recurrent_fw_C_metal
)
from xlstm_metal.mlx_jit.blocks.mlstm.forward.mlstm_chunkwise_parallel_fw_Hintra import (
    mlstm_chunkwise_parallel_fw_Hintra_metal
)


class mLSTMNeuron(nn.Module):
    """
    Single mLSTM neuron for NCPS wiring.
    
    This is the atomic unit - a single head of matrix LSTM.
    Multiple neurons are wired together to form layers.
    
    Args:
        qk_dim: Query/key dimension for this neuron
        v_dim: Value dimension for this neuron
        use_bias: Whether to use bias in projections
        eps: Numerical stability epsilon
        kernel_mode: Execution strategy ('sequential', 'parallel', 'chunkwise')
        chunk_size: Chunk size for chunkwise mode (default 64)
    """
    
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        use_bias: bool = False,
        eps: float = 1e-6,
        kernel_mode: str = "chunkwise",
        chunk_size: int = 64,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.eps = eps
        self.kernel_mode = kernel_mode
        self.chunk_size = chunk_size
        
        # This neuron maintains state (C, n, m)
        # State shapes: [DHQK, DHHV], [DHQK], scalar
        self._C = None  # Covariance matrix
        self._n = None  # Normalizer
        self._m = None  # Running max
        
    def reset_state(self, batch_size: int = 1):
        """Initialize/reset neuron state."""
        self._C = mx.zeros((batch_size, self.qk_dim, self.v_dim), dtype=mx.float32)
        self._n = mx.zeros((batch_size, self.qk_dim), dtype=mx.float32)
        self._m = mx.zeros((batch_size,), dtype=mx.float32)
        
    def __call__(
        self,
        q: mx.array,  # [B, DHQK]
        k: mx.array,  # [B, DHQK]
        v: mx.array,  # [B, DHHV]
        i_preact: mx.array,  # [B] - input gate pre-activation
        f_preact: mx.array,  # [B] - forget gate pre-activation
    ) -> mx.array:
        """
        Single-step mLSTM neuron update.
        
        Args:
            q: Query [B, DHQK]
            k: Key [B, DHQK]
            v: Value [B, DHHV]
            i_preact: Input gate pre-activation [B]
            f_preact: Forget gate pre-activation [B]
            
        Returns:
            h: Output [B, DHHV]
        """
        B = q.shape[0]
        
        # Initialize state if needed
        if self._C is None:
            self.reset_state(B)
            
        # Ensure states are float32
        C_state = self._C.astype(mx.float32)
        n_state = self._n.astype(mx.float32)
        m_state = self._m.astype(mx.float32)
        
        # Apply logsigmoid to forget gate
        one = mx.array(1.0, dtype=f_preact.dtype)
        f_log = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(f_preact)))))
        
        # Update running max: m_t = max(f_log + m_{t-1}, i_t)
        m_new = mx.maximum(mx.add(f_log, m_state), i_preact)
        
        # Normalized exponential gates
        f_exp = mx.exp(mx.subtract(mx.add(f_log, m_state), m_new))
        i_exp = mx.exp(mx.subtract(i_preact, m_new))
        
        # Expand for broadcasting
        i_exp_expanded = i_exp[:, None, None]  # [B, 1, 1]
        f_exp_expanded = f_exp[:, None, None]  # [B, 1, 1]
        
        # Update covariance: C_t = f * C_{t-1} + i * (k ⊗ v)
        kv_outer = k[:, :, None] * v[:, None, :]  # [B, DHQK, DHHV]
        C_new = f_exp_expanded * C_state + i_exp_expanded * kv_outer
        
        # Update normalizer: n_t = f * n_{t-1} + i * k
        n_new = f_exp[:, None] * n_state + i_exp[:, None] * k
        
        # Scale query by 1/√d_qk
        q_scaled = q * mx.rsqrt(mx.array(self.qk_dim, dtype=q.dtype))
        
        # Compute output: h_t = (q^T @ C_t) / (|q·n_t| + exp(-m_t) + eps)
        h_num = mx.matmul(C_new.transpose(0, 2, 1), q_scaled[:, :, None]).squeeze(-1)
        
        qn_dot = mx.sum(n_new * q_scaled, axis=-1, keepdims=True)
        max_val = mx.exp(mx.negative(m_new))[:, None]
        h_den = mx.maximum(mx.abs(qn_dot), max_val) + self.eps
        
        h = h_num / h_den
        
        # Update state
        self._C = C_new
        self._n = n_new
        self._m = m_new
        
        return h.astype(q.dtype)


class mLSTMChunkwiseCell(nn.Module):
    """
    Multi-head mLSTM cell using chunkwise parallel processing.
    
    This processes sequences in parallel chunks using Metal kernels.
    Suitable for training and long sequences.
    
    Args:
        input_size: Input dimension
        num_heads: Number of mLSTM neurons (heads)
        qk_dim_per_head: QK dimension per neuron
        v_dim_per_head: V dimension per neuron
        chunk_size: Chunk size for parallel kernel
        use_bias: Whether to use bias
        eps: Numerical stability epsilon
    """
    
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        qk_dim_per_head: int,
        v_dim_per_head: int,
        chunk_size: int = 64,
        use_bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim_per_head
        self.v_dim_per_head = v_dim_per_head
        self.chunk_size = chunk_size
        self.eps = eps
        
        qk_dim = num_heads * qk_dim_per_head
        v_dim = num_heads * v_dim_per_head
        
        # Projections
        self.q_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.v_proj = nn.Linear(input_size, v_dim, bias=use_bias)
        self.igate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.ogate_proj = nn.Linear(input_size, v_dim, bias=use_bias)
        
        # Norm and output
        self.norm = nn.RMSNorm(v_dim_per_head, eps=eps)
        self.out_proj = nn.Linear(v_dim, input_size, bias=use_bias)
        
    def __call__(self, x: mx.array, state=None) -> Tuple[mx.array, Tuple]:
        """Process sequence using chunkwise parallel Metal kernel."""
        B, S, _ = x.shape
        
        # Project
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.qk_dim_per_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.qk_dim_per_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.v_dim_per_head).transpose(0, 2, 1, 3)
        i_preact = self.igate_proj(x).transpose(0, 2, 1)
        f_preact = self.fgate_proj(x).transpose(0, 2, 1)
        
        # Initialize state
        c_initial = state[0] if state else None
        n_initial = state[1] if state else None
        m_initial = state[2] if state else None
        
        # Recurrent phase - compute inter-chunk states
        matC_states, vecN_states, scaMinter_states = mlstm_chunkwise_recurrent_fw_C_metal(
            matK=k.astype(mx.float32),
            matV=v.astype(mx.float32),
            vecF=f_preact.astype(mx.float32),
            vecI=i_preact.astype(mx.float32),
            matC_initial=c_initial.astype(mx.float32) if c_initial is not None else None,
            vecN_initial=n_initial.astype(mx.float32) if n_initial is not None else None,
            scaMinter_initial=m_initial.astype(mx.float32) if m_initial is not None else None,
            NC=(S + self.chunk_size - 1) // self.chunk_size,
            L=self.chunk_size,
            siz_b_DHQK=16,
            siz_b_DHHV=16,
            save_states_every_nth_chunk=1,
        )
        
        # Parallel phase - compute outputs within chunks
        NC = (S + self.chunk_size - 1) // self.chunk_size
        vecI_chunked = i_preact.reshape(B, self.num_heads, NC, self.chunk_size)
        vecF_chunked = f_preact.reshape(B, self.num_heads, NC, self.chunk_size)
        
        one = mx.array(1.0, dtype=vecF_chunked.dtype)
        vecF_logsig = mx.negative(mx.log(mx.add(one, mx.exp(mx.negative(vecF_chunked)))))
        vecB = mx.cumsum(vecF_logsig, axis=-1)
        
        qk_scale = 1.0 / (self.qk_dim_per_head ** 0.5)
        
        matHout, vecNout, vecMout = mlstm_chunkwise_parallel_fw_Hintra_metal(
            matQ=q.astype(mx.float32),
            matK=k.astype(mx.float32),
            matV=v.astype(mx.float32),
            matC_states=matC_states.astype(mx.float32),
            vecN_states=vecN_states.astype(mx.float32),
            scaMinter_states=scaMinter_states.astype(mx.float32),
            vecI=vecI_chunked.astype(mx.float32),
            vecB=vecB.astype(mx.float32),
            NC=NC,
            L=self.chunk_size,
            qk_scale=qk_scale,
            siz_b_LQ=8,
            siz_b_LKV=8,
            siz_b_DHQK=8,
            siz_b_DHHV=8,
            eps=self.eps,
            minimum_max_val=-10.0,
        )
        
        mx.eval(matHout)
        
        # Transpose back
        h = matHout.transpose(0, 2, 1, 3)  # [B, S, NH, V_DH]
        
        # Norm and output gate
        h_norm = self.norm(h)
        o_gate = mx.sigmoid(self.ogate_proj(x)).reshape(B, S, self.num_heads, self.v_dim_per_head)
        h_gated = h_norm * o_gate
        
        # Output projection
        h_flat = h_gated.reshape(B, S, self.num_heads * self.v_dim_per_head)
        output = self.out_proj(h_flat)
        
        # Return with final state
        new_state = (matC_states[:, :, -1], vecN_states[:, :, -1], scaMinter_states[:, :, -1])
        return output, new_state


__all__ = ['mLSTMNeuron', 'mLSTMChunkwiseCell']
