"""
mLSTM Block for xLSTM-7B

Implements the complete mLSTM layer matching the HuggingFace xLSTM-7b weight structure.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .components import MultiHeadLayerNorm, RMSNorm, soft_cap


@dataclass
class mLSTMConfig:
    """
    Configuration for mLSTM block matching xLSTM-7B.

    From xlstm_7b_model/config.json:
        embedding_dim: 4096
        num_heads: 8
        head_dim: 512  (v_dim / num_heads)
        qk_dim_factor: 0.5  (qk_dim = 2048)
        v_dim_factor: 1.0   (v_dim = 4096)
        gate_soft_cap: 15.0

    Kernel configuration:
        chunk_size: 64  (chunk size for chunkwise parallel processing)
        siz_b_DHQK: 16  (Metal threadgroup size for recurrent kernel)
        siz_b_DHHV: 16  (Metal threadgroup size for recurrent kernel)
        siz_b_LQ: 8     (Metal threadgroup size for parallel kernel)
        siz_b_LKV: 8    (Metal threadgroup size for parallel kernel)
        siz_b_DHQK_parallel: 8  (Metal threadgroup size for parallel kernel)
        siz_b_DHHV_parallel: 8  (Metal threadgroup size for parallel kernel)
    """
    embedding_dim: int = 4096
    num_heads: int = 8
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True

    # Kernel performance parameters
    chunk_size: int = 64
    max_chunk_size: int = 64  # Metal kernel buffer limit
    siz_b_DHQK: int = 16
    siz_b_DHHV: int = 16
    siz_b_LQ: int = 8
    siz_b_LKV: int = 8
    siz_b_DHQK_parallel: int = 8
    siz_b_DHHV_parallel: int = 8

    def __post_init__(self):
        self.qk_dim = int(self.embedding_dim * self.qk_dim_factor)
        self.v_dim = int(self.embedding_dim * self.v_dim_factor)
        self.head_dim = self.v_dim // self.num_heads
        self.qk_head_dim = self.qk_dim // self.num_heads

        # Validate chunk_size
        if self.chunk_size > self.max_chunk_size:
            raise ValueError(f"chunk_size={self.chunk_size} exceeds Metal kernel buffer limit of {self.max_chunk_size}!")


class mLSTMLayer(nn.Module):
    """
    mLSTM Layer matching xLSTM-7B weight structure.

    Weight structure (from model.safetensors.index.json):
        q.weight: [2048, 4096]
        k.weight: [2048, 4096]
        v.weight: [4096, 4096]
        igate_preact.weight: [8, 4096]
        igate_preact.bias: [8]
        fgate_preact.weight: [8, 4096]
        fgate_preact.bias: [8]
        ogate_preact.weight: [4096, 4096]
        multihead_norm.weight: [8, 512]
        out_proj.weight: [4096, 4096]
    """

    def __init__(self, config: mLSTMConfig):
        super().__init__()
        self.config = config

        # QKV projections
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

        # Gate projections (PER-HEAD!)
        # Note: igate and fgate have bias=True even when use_bias=False
        self.igate_preact = nn.Linear(config.embedding_dim, config.num_heads)
        self.fgate_preact = nn.Linear(config.embedding_dim, config.num_heads)
        self.ogate_preact = nn.Linear(
            config.embedding_dim,
            config.v_dim,
            bias=config.use_bias
        )

        # Multi-head layer normalization (per-head, not standard!)
        self.multihead_norm = MultiHeadLayerNorm(num_heads=config.num_heads, head_dim=config.head_dim,
                                                 eps=config.norm_eps, use_bias=config.use_bias,
                                                 force_float32_reductions=config.norm_reduction_force_float32)

        # Output projection
        self.out_proj = nn.Linear(
            config.v_dim,
            config.embedding_dim,
            bias=config.use_bias
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass

        Args:
            x: Input tensor [B, S, embedding_dim]
            state: Optional state tuple (c, n, m)
                c: Covariance [B, num_heads, head_dim, qk_head_dim]
                n: Normalizer [B, num_heads, qk_head_dim]
                m: Running max [B, num_heads]

        Returns:
            y: Output tensor [B, S, embedding_dim]
            state: Updated state tuple if return_last_states else None

        Note: head_dim = v_dim // num_heads (output dim per head)
              qk_head_dim = qk_dim // num_heads (query/key dim per head)
        """
        B, S, D = x.shape
        assert D == self.config.embedding_dim

        # 1. QKV projections
        q = self.q(x)  # [B, S, qk_dim]
        k = self.k(x)  # [B, S, qk_dim]
        v = self.v(x)  # [B, S, v_dim]

        # 2. Gate projections
        i_preact = self.igate_preact(x)  # [B, S, num_heads]
        f_preact = self.fgate_preact(x)  # [B, S, num_heads]
        o_preact = self.ogate_preact(x)  # [B, S, v_dim]

        # 3. Soft-cap gates (15.0)
        i_preact = soft_cap(i_preact, self.config.gate_soft_cap)
        f_preact = soft_cap(f_preact, self.config.gate_soft_cap)

        # 4. Reshape for multi-head processing
        # q, k: [B, S, qk_dim] -> [B, num_heads, S, qk_head_dim]
        q = q.reshape(B, S, self.config.num_heads, self.config.qk_head_dim)
        q = q.transpose(0, 2, 1, 3)  # [B, num_heads, S, qk_head_dim]

        k = k.reshape(B, S, self.config.num_heads, self.config.qk_head_dim)
        k = k.transpose(0, 2, 1, 3)  # [B, num_heads, S, qk_head_dim]

        # v: [B, S, v_dim] -> [B, num_heads, S, head_dim]
        v = v.reshape(B, S, self.config.num_heads, self.config.head_dim)
        v = v.transpose(0, 2, 1, 3)  # [B, num_heads, S, head_dim]

        # i_preact, f_preact: [B, S, num_heads] -> [B, num_heads, S]
        i_preact = i_preact.transpose(0, 2, 1)  # [B, num_heads, S]
        f_preact = f_preact.transpose(0, 2, 1)  # [B, num_heads, S]

        # 5. Extract initial states
        if state is None:
            c_initial = None
            n_initial = None
            m_initial = None
        else:
            c_initial, n_initial, m_initial = state

        # 6. mLSTM backend - select kernel based on sequence length (matches PyTorch)
        #    - S == 1: single step recurrent
        #    - 1 < S < chunk_size: sequential (loop over recurrent steps)
        #    - S >= chunk_size: chunkwise parallel
        if S == 1:
            # Single token: use recurrent step
            from .kernel import mlstm_recurrent_step

            # Squeeze S dimension for recurrent step
            q_t = q[:, :, 0, :]  # [B, NH, QK_DH]
            k_t = k[:, :, 0, :]  # [B, NH, QK_DH]
            v_t = v[:, :, 0, :]  # [B, NH, V_DH]
            i_t = i_preact[:, :, 0]  # [B, NH]
            f_t = f_preact[:, :, 0]  # [B, NH]

            # Initialize states if needed
            # CRITICAL: States must be float32 for numerical stability
            if c_initial is None:
                NH = q.shape[1]
                QK_DH = q.shape[3]
                V_DH = v.shape[3]
                c_initial = mx.zeros((B, NH, QK_DH, V_DH))
                n_initial = mx.zeros((B, NH, QK_DH))
                m_initial = mx.zeros((B, NH))

            # Recurrent step
            h_t, c_new, n_new, m_new = mlstm_recurrent_step(
                q=q_t,
                k=k_t,
                v=v_t,
                i_preact=i_t,
                f_preact=f_t,
                c_state=c_initial,
                n_state=n_initial,
                m_state=m_initial,
                eps=self.config.eps
            )

            # Add back S dimension
            h = h_t[:, :, None, :]  # [B, NH, 1, V_DH]
            new_state = (c_new, n_new, m_new) if self.config.return_last_states else None
        elif S < self.config.chunk_size:
            # Short sequence (1 < S < chunk_size): use sequential loop
            # This matches PyTorch's wrap_chunkwise_arbitrary_sequence_length behavior
            from .kernel import mlstm_sequential
            h, new_state = mlstm_sequential(
                q=q,
                k=k,
                v=v,
                i_preact=i_preact,
                f_preact=f_preact,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
                eps=self.config.eps,
                return_last_states=self.config.return_last_states
            )
        else:
            # Long sequence (S >= chunk_size): use chunkwise parallel kernel
            from .kernel import mlstm_chunkwise
            h, new_state = mlstm_chunkwise(
                q=q,
                k=k,
                v=v,
                i_preact=i_preact,
                f_preact=f_preact,
                chunk_size=self.config.chunk_size,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
                eps=self.config.eps,
                return_last_states=self.config.return_last_states,
                siz_b_DHQK=self.config.siz_b_DHQK,
                siz_b_DHHV=self.config.siz_b_DHHV,
                siz_b_LQ=self.config.siz_b_LQ,
                siz_b_LKV=self.config.siz_b_LKV,
                siz_b_DHQK_parallel=self.config.siz_b_DHQK_parallel,
                siz_b_DHHV_parallel=self.config.siz_b_DHHV_parallel,
                max_chunk_size=self.config.max_chunk_size
            )
        # h: [B, num_heads, S, head_dim]

        # 7. Transpose back and reshape
        h = h.transpose(0, 2, 1, 3)  # [B, S, num_heads, head_dim]

        # 8. Multi-head normalization
        h_norm = self.multihead_norm(h)  # [B, S, num_heads, head_dim]

        # 9. Reshape for output projection
        h_norm = h_norm.reshape(B, S, self.config.v_dim)  # [B, S, v_dim]

        # 10. Apply output gate (sigmoid)
        h_out = mx.multiply(mx.sigmoid(o_preact), h_norm)  # [B, S, v_dim]

        # 11. Output projection
        y = self.out_proj(h_out)  # [B, S, embedding_dim]

        return y, new_state


class mLSTMBlock(nn.Module):
    """
    Complete mLSTM block with pre-normalization.

    Matches the structure:
        x -> RMSNorm -> mLSTMLayer -> (+) residual
    """

    def __init__(self, config: mLSTMConfig):
        super().__init__()
        self.config = config

        self.norm_mlstm = RMSNorm(num_features=config.embedding_dim, eps=config.norm_eps, use_bias=config.use_bias,
                                  force_float32_reductions=config.norm_reduction_force_float32)

        self.mlstm_layer = mLSTMLayer(config)

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass with pre-normalization and residual

        Args:
            x: Input [B, S, embedding_dim]
            state: Optional mLSTM state

        Returns:
            x_out: Output with residual [B, S, embedding_dim]
            state: Updated state
        """
        # Pre-normalization
        x_norm = self.norm_mlstm(x)

        # mLSTM layer
        x_mlstm, state = self.mlstm_layer(x_norm, state)

        # Residual connection
        x_out = mx.add(x, x_mlstm)

        return x_out, state
