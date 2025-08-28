"""FeedForward layer implementation."""

import torch
from torch import nn

# Import utilities
import importlib.util
spec = importlib.util.find_spec("xlstm")
if spec is not None:
    from xlstm.xlstm_large.utils import round_up_to_next_multiple_of
else:
    def round_up_to_next_multiple_of(x, multiple):
        return ((x + multiple - 1) // multiple) * multiple

class FeedForward(nn.Module):
    """FeedForward layer with SwiGLU activation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.up_proj_dim = round_up_to_next_multiple_of(
            config.embedding_dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )
        
        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(config.embedding_dim, self.up_proj_dim, bias=config.use_bias)
            self.proj_up = nn.Linear(config.embedding_dim, self.up_proj_dim, bias=config.use_bias)
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(config.embedding_dim, 2 * self.up_proj_dim, bias=config.use_bias)
        
        self.proj_down = nn.Linear(self.up_proj_dim, config.embedding_dim, bias=config.use_bias)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z
        
        y = self.proj_down(x)
        return y