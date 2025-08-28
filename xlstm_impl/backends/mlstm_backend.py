"""mLSTM backend implementation."""

from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Tuple, Literal

# Type definitions
DtypeType = Literal["float32", "float16", "bfloat16"]
BackendModeType = Literal["train", "train_with_padding", "inference"]
KernelType = Literal["native", "metal", "cuda"]

@dataclass
class mLSTMBackendConfig:
    """Configuration for mLSTM backend."""
    chunkwise_kernel: KernelType = "native"
    sequence_kernel: KernelType = "native"
    step_kernel: KernelType = "native"
    mode: BackendModeType = "train"
    chunk_size: int = 64
    return_last_states: bool = False
    autocast_kernel_dtype: DtypeType = "float16"
    eps: float = 1e-6
    inference_state_dtype: DtypeType = "float32"

class mLSTMBackend(nn.Module):
    """mLSTM backend that works with Metal, CUDA, or CPU."""
    def __init__(self, config: mLSTMBackendConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.return_last_states = config.return_last_states or config.mode == "inference"
    
    def _init_state(self, B: int, NH: int, DH_v: int, DH_k: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize state tensors.
        
        Args:
            B: Batch size
            NH: Number of heads
            DH_v: Value dimension per head
            DH_k: Key dimension per head
            device: Device to place tensors on
        """
        c_0 = torch.zeros(B, NH, DH_v, DH_k, device=device, dtype=torch.float32)
        n_0 = torch.ones(B, NH, DH_k, device=device, dtype=torch.float32) 
        m_0 = torch.zeros(B, NH, device=device, dtype=torch.float32)
        return c_0, n_0, m_0
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                i: torch.Tensor, f: torch.Tensor,
                c_initial: Optional[torch.Tensor] = None,
                n_initial: Optional[torch.Tensor] = None, 
                m_initial: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Forward pass for mLSTM.
        
        Args:
            q: Query tensor [B, NH, S, DH_q]
            k: Key tensor [B, NH, S, DH_k]
            v: Value tensor [B, NH, S, DH_v]
            i: Input gate [B, NH, S]
            f: Forget gate [B, NH, S]
            c_initial: Initial covariance state
            n_initial: Initial normalizer state
            m_initial: Initial max state
            
        Returns:
            h: Hidden states [B, NH, S, DH_v]
            state: Optional state tuple (c, n, m)
        """
        B, NH, S, DH_q = q.shape
        DH_v = v.shape[-1]
        DH_k = k.shape[-1]
        
        # Initialize states if not provided
        if c_initial is None or n_initial is None or m_initial is None:
            c_initial, n_initial, m_initial = self._init_state(B, NH, DH_v, DH_k, q.device)
        
        # Process sequence  
        h_out = torch.zeros(B, NH, S, DH_v, device=q.device, dtype=q.dtype)
        
        c_state = c_initial
        n_state = n_initial
        m_state = m_initial
        
        for t in range(S):
            # Get current timestep
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            i_t = i[:, :, t]
            f_t = f[:, :, t]
            
            # Exponential gating
            m_new = torch.maximum(f_t + m_state, i_t)
            i_exp = torch.exp(i_t - m_new)
            f_exp = torch.exp(f_t - m_new + m_state)
            
            # Expand gates
            i_expanded = i_exp.unsqueeze(-1).unsqueeze(-1)
            f_expanded = f_exp.unsqueeze(-1).unsqueeze(-1)
            
            # Update covariance matrix
            v_expanded = v_t.unsqueeze(-1)
            k_expanded = k_t.unsqueeze(-2)
            vk_outer = torch.matmul(v_expanded, k_expanded)
            
            c_state = f_expanded * c_state + i_expanded * vk_outer
            
            # Update normalizer
            f_n = f_exp.unsqueeze(-1)
            i_n = i_exp.unsqueeze(-1)
            n_state = f_n * n_state + i_n * k_t
            
            # Compute output
            q_expanded = q_t.unsqueeze(-1)
            h_num = torch.matmul(c_state, q_expanded).squeeze(-1)
            h_den = torch.sum(n_state * q_t, dim=-1, keepdim=True).clamp(min=self.config.eps)
            
            h_out[:, :, t, :] = h_num / h_den
            
            m_state = m_new
        
        if self.return_last_states:
            return h_out, (c_state, n_state, m_state)
        else:
            return h_out, None