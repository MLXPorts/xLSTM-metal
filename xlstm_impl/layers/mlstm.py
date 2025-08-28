"""mLSTM layer implementations."""

from dataclasses import dataclass, field
import torch
from torch import nn
from typing import Optional, Tuple, Literal

# Import from installed xlstm package or use fallbacks
import importlib.util
spec = importlib.util.find_spec("xlstm")
if spec is not None:
    from xlstm.xlstm_large.components import MultiHeadLayerNorm, RMSNorm, soft_cap
    from xlstm.xlstm_large.utils import round_up_to_next_multiple_of
else:
    # Use implementations from models
    from ..models.xlstm import RMSNorm, soft_cap
    
    def round_up_to_next_multiple_of(x, multiple):
        return ((x + multiple - 1) // multiple) * multiple
    
    # Simple MultiHeadLayerNorm implementation
    class MultiHeadLayerNorm(torch.nn.Module):
        def __init__(self, num_heads, head_dim, eps=1e-6, use_weight=True, use_bias=False, force_float32_reductions=True):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.norm = torch.nn.LayerNorm(num_heads * head_dim, eps=eps)
            
        def forward(self, x):
            B, S, NH, DH = x.shape
            x = x.reshape(B, S, -1)
            return self.norm(x)
from ..backends.mlstm_backend import mLSTMBackend, mLSTMBackendConfig
from .feedforward import FeedForward

# Type definitions
mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]
WeightModeType = Literal["single", "fused"]

@dataclass
class mLSTMLayerConfig:
    """Configuration for mLSTM layer."""
    embedding_dim: int
    num_heads: int
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    mlstm_backend: mLSTMBackendConfig = field(default_factory=mLSTMBackendConfig)
    weight_mode: WeightModeType = "single"

class mLSTMLayer(nn.Module):
    """mLSTM layer implementation."""
    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config
        
        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)
        
        if self.config.weight_mode == "single":
            self.q = nn.Linear(config.embedding_dim, self.qk_dim, bias=config.use_bias)
            self.k = nn.Linear(config.embedding_dim, self.qk_dim, bias=config.use_bias)
            self.v = nn.Linear(config.embedding_dim, self.v_dim, bias=config.use_bias)
            self.ogate_preact = nn.Linear(config.embedding_dim, self.v_dim, bias=config.use_bias)
            self.igate_preact = nn.Linear(config.embedding_dim, config.num_heads, bias=True)
            self.fgate_preact = nn.Linear(config.embedding_dim, config.num_heads, bias=True)
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                config.embedding_dim,
                2 * self.qk_dim + 2 * self.v_dim,
                bias=config.use_bias
            )
            self.ifgate_preact = nn.Linear(config.embedding_dim, 2 * config.num_heads, bias=True)
        
        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)
        
        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=config.num_heads,
            head_dim=self.v_dim // config.num_heads,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(self.v_dim, config.embedding_dim, bias=config.use_bias)
    
    def forward(
        self, x: torch.Tensor, state: mLSTMLayerStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape
        
        if self.config.weight_mode == "single":
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            o_preact = self.ogate_preact(x)
            i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
            f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)
        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (self.qk_dim, 2 * self.qk_dim, 2 * self.qk_dim + self.v_dim),
                dim=-1,
            )
            if_preact = soft_cap(self.ifgate_preact(x), cap_value=self.config.gate_soft_cap)
            i_preact, f_preact = torch.tensor_split(if_preact, (self.config.num_heads,), dim=-1)
        
        # Reshape for multi-head processing
        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state
        
        h, state = self.mlstm_backend(
            q=q, k=k, v=v, i=i_preact, f=f_preact,
            c_initial=c_initial, n_initial=n_initial, m_initial=m_initial
        )
        
        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)
        
        h_out = self.ogate_act_fn(o_preact) * h_norm
        
        y = self.out_proj(h_out)
        return y, state

class mLSTMBlock(nn.Module):
    """mLSTM block with residual connections and normalization."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_mlstm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.mlstm_layer = mLSTMLayer(
            mLSTMLayerConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                norm_reduction_force_float32=config.norm_reduction_force_float32,
                qk_dim_factor=config.qk_dim_factor,
                v_dim_factor=config.v_dim_factor,
                gate_soft_cap=config.gate_soft_cap,
                weight_mode=config.weight_mode,
                mlstm_backend=mLSTMBackendConfig(
                    chunkwise_kernel=config.chunkwise_kernel,
                    sequence_kernel=config.sequence_kernel,
                    step_kernel=config.step_kernel,
                    mode=config.mode,
                    chunk_size=config.chunk_size,
                    return_last_states=config.return_last_states,
                    autocast_kernel_dtype=config.autocast_kernel_dtype,
                    eps=config.eps,
                    inference_state_dtype=config.inference_state_dtype,
                ),
            )
        )
        self.norm_ffn = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.ffn = FeedForward(config)
    
    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm
        
        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn
        
        return x, state