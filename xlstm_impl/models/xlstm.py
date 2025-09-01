"""Main xLSTM model implementation."""

from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Union, Literal

# Import from installed xlstm package
import sys
import importlib.util
spec = importlib.util.find_spec("xlstm")
if spec is not None:
    from xlstm.xlstm_large.components import RMSNorm, soft_cap
    from xlstm.xlstm_large.generate import generate_tokens, get_sampling_fn
else:
    # Fallback implementations
    import torch
    from torch import nn
    
    def soft_cap(values, cap_value=None):
        if cap_value is None:
            return values
        return cap_value * torch.tanh(values / cap_value)
    
    class RMSNorm(nn.Module):
        def __init__(self, num_features, eps=1e-6, use_weight=True, use_bias=False, force_float32_reductions=True):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(num_features)) if use_weight else None
            self.bias = nn.Parameter(torch.zeros(num_features)) if use_bias else None
            
        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            if self.weight is not None:
                norm = norm * self.weight
            if self.bias is not None:
                norm = norm + self.bias
            return norm.to(dtype)
    
    # Placeholder for generate functions
    def generate_tokens(*args, **kwargs):
        raise NotImplementedError("generate_tokens not available without xlstm package")
    
    def get_sampling_fn(*args, **kwargs):
        raise NotImplementedError("get_sampling_fn not available without xlstm package")
from ..layers.mlstm import mLSTMBlock
from ..backends.mlstm_backend import mLSTMBackendConfig
from ..utils.device import DEVICE

# Type definitions
mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]
WeightModeType = Literal["single", "fused"]

# Backend type definitions  
ChunkwiseKernelType = Literal["native", "metal", "cuda"]
SequenceKernelType = Literal["native", "metal", "cuda"]
StepKernelType = Literal["native", "metal", "cuda"]
DtypeType = Literal["float32", "float16", "bfloat16"]
BackendModeType = Literal["train", "train_with_padding", "inference"]

@dataclass
class xLSTMLargeConfig:
    """Configuration for xLSTM model."""
    embedding_dim: int
    num_heads: int
    num_blocks: int
    vocab_size: int
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    add_out_norm: bool = True
    
    # mlstm layer
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    
    # mlstm backend - auto-detect best kernel
    chunkwise_kernel: ChunkwiseKernelType = "metal" if DEVICE == "mps" else ("cuda" if DEVICE == "cuda" else "native")
    sequence_kernel: SequenceKernelType = "metal" if DEVICE == "mps" else ("cuda" if DEVICE == "cuda" else "native")
    step_kernel: StepKernelType = "metal" if DEVICE == "mps" else ("cuda" if DEVICE == "cuda" else "native")
    mode: BackendModeType = "train"
    chunk_size: int = 64
    return_last_states: bool = False
    autocast_kernel_dtype: DtypeType = "float16" if DEVICE == "mps" else "bfloat16"
    eps: float = 1e-6
    inference_state_dtype: DtypeType = "float32"
    
    # feedforward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64
    
    # capping
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    
    weight_mode: WeightModeType = "single"

class xLSTMLargeBlockStack(nn.Module):
    """Block stack for xLSTM."""
    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config
        
        self.blocks = nn.ModuleList(
            [mLSTMBlock(config) for _ in range(config.num_blocks)]
        )
        
        if self.config.add_out_norm:
            self.out_norm = RMSNorm(
                num_features=config.embedding_dim,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
        else:
            self.out_norm = nn.Identity()
    
    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        if state is None:
            state = {i: None for i in range(len(self.blocks))}
        
        for i, block in enumerate(self.blocks):
            block_state = state[i]
            x, block_state_new = block(x, block_state)
            
            if block_state is None:
                state[i] = block_state_new
            else:
                # Update state in place
                for state_idx in range(len(block_state)):
                    state[i][state_idx].copy_(block_state_new[state_idx])
        
        x = self.out_norm(x)
        
        return x, state

class xLSTMLarge(nn.Module):
    """xLSTM model with automatic device optimization."""
    config_class = xLSTMLargeConfig
    
    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.backbone = xLSTMLargeBlockStack(config)
        
        self.lm_head = nn.Linear(
            in_features=config.embedding_dim, out_features=config.vocab_size, bias=False
        )
    
    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, mLSTMStateType]]:
        """Forward pass of the model."""
        assert x.ndim == 2, f"Input must have shape [B, S], got {x.shape}"
        B, S = x.shape
        
        x = self.embedding(x)
        
        x, state = self.backbone(x, state)
        
        logits = self.lm_head(x)
        logits_capped = soft_cap(logits, self.config.output_logit_soft_cap)
        
        if self.config.return_last_states:
            return logits_capped, state
        else:
            return logits_capped
    
    def generate(
        self,
        prefill_tokens: torch.Tensor,
        max_length: int,
        sampling_type: str = "greedy",
        state: mLSTMStateType | None = None,
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        """Generate tokens from the model."""
        sampling_fn = get_sampling_fn(sampling_type)
        tokens, state = generate_tokens(
            llm_forward=self.forward,
            prefill_tokens=prefill_tokens,
            max_length=max_length,
            token_sample_fn=sampling_fn,
            state=state,
            device=str(self.embedding.weight.device),
        )
        return tokens, state