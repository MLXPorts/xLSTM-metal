"""mLSTM cells for NCPS wiring.

NCPS-compatible mLSTM cells with Metal kernel acceleration.
Following NCPS patterns: cells process single timesteps, wrappers handle sequences.
"""

from __future__ import annotations
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Direct Metal kernel imports
from xlstm_metal.mlx_jit.blocks.mlstm.forward import (
    mlstm_chunkwise_recurrent_fw_C_metal,
    mlstm_chunkwise_parallel_fw_Hintra_metal
)


class mLSTMCell(nn.Module):
    """
    Multi-head mLSTM cell using chunkwise parallel Metal kernels.
    
    NCPS-compatible cell following the LTCCell pattern:
    - Parameters are direct attributes (tracked by nn.Module)
    - State passed in/out of __call__
    - Single timestep processing (wrapper handles sequences)
    
    Args:
        input_size: Input dimension
        num_heads: Number of parallel mLSTM heads
        qk_dim_per_head: Query/Key dimension per head
        v_dim_per_head: Value dimension per head
        chunk_size: Chunk size for parallel Metal kernel
        use_bias: Whether to use bias in projections
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
        
        # Store dimensions
        self._input_size = input_size
        self._num_heads = num_heads
        self._qk_dim_per_head = qk_dim_per_head
        self._v_dim_per_head = v_dim_per_head
        self._chunk_size = chunk_size
        self._eps = eps

        qk_dim = num_heads * qk_dim_per_head
        v_dim = num_heads * v_dim_per_head

        # Trainable parameters (direct attributes, tracked by nn.Module)
        self.q_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_size, qk_dim, bias=use_bias)
        self.v_proj = nn.Linear(input_size, v_dim, bias=use_bias)
        self.igate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(input_size, num_heads, bias=True)
        self.ogate_proj = nn.Linear(input_size, v_dim, bias=use_bias)
        self.norm = nn.RMSNorm(v_dim_per_head, eps=eps)
        self.out_proj = nn.Linear(v_dim, input_size, bias=use_bias)
    
    @property
    def state_size(self) -> Tuple[int, int, int]:
        """Return state shape: (C, n, m) dimensions."""
        return (
            self._num_heads * self._qk_dim_per_head * self._v_dim_per_head,
            self._num_heads * self._qk_dim_per_head,
            self._num_heads
        )
    
    @property
    def output_size(self) -> int:
        """Return output dimension."""
        return self._input_size

    def __call__(self, x: mx.array, state: Optional[Tuple] = None) -> Tuple[mx.array, Tuple]:
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


__all__ = ['mLSTMModel', 'mLSTMChunkwiseCell']
