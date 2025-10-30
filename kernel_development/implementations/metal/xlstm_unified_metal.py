
"""
Unified xLSTM implementation with optimized Metal Performance Shaders backend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
import math
from dataclasses import dataclass


@dataclass
class xLSTMConfig:
    vocab_size: int = 50257
    num_layers: int = 6
    d_model: int = 512
    signature: Tuple[int, ...] = (1, 1)
    inp_dim: int = 128
    head_dim: int = 32
    head_num: int = 4
    mlstm_proj_factor: float = 2.0
    slstm_proj_factor: float = 1.333
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    dropout: float = 0.1
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    use_bias: bool = False
    causal_conv_kernel: int = 4
    device: str = "cpu"


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        """

        :param hidden_states:
        :return:
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class mLSTMBlock(nn.Module):
    def __init__(self, config: xLSTMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        proj_dim = int(config.d_model * config.mlstm_proj_factor)
        self.head_dim = config.head_dim
        self.num_heads = config.head_num
        
        # Input projections
        self.q_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=config.use_bias)
        
        # Gates
        self.i_proj = nn.Linear(config.d_model, self.num_heads, bias=config.use_bias)
        self.f_proj = nn.Linear(config.d_model, self.num_heads, bias=config.use_bias)
        self.o_proj = nn.Linear(config.d_model, self.num_heads, bias=config.use_bias)
        
        # Output projection
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=config.use_bias)
        
        # Layer norm
        if config.norm_type == "rmsnorm":
            self.layer_norm = RMSNorm(config.d_model, config.norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
    
    def soft_cap(self, x: torch.Tensor, cap_value: float) -> torch.Tensor:
        """

        :param x:
        :param cap_value:
        :return:
        """
        return cap_value * torch.tanh(x / cap_value)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x:
        :param hidden_state:
        :return:
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        x = self.layer_norm(x)
        
        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Gates
        i_gate = torch.sigmoid(self.soft_cap(self.i_proj(x), self.config.gate_soft_cap))
        f_gate = torch.sigmoid(self.soft_cap(self.f_proj(x), self.config.gate_soft_cap))
        o_gate = torch.sigmoid(self.soft_cap(self.o_proj(x), self.config.gate_soft_cap))
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                dtype=x.dtype, device=x.device
            )
        
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t]  # [batch_size, num_heads, head_dim]
            k_t = k[:, t]  # [batch_size, num_heads, head_dim]
            v_t = v[:, t]  # [batch_size, num_heads, head_dim]
            i_t = i_gate[:, t]  # [batch_size, num_heads]
            f_t = f_gate[:, t]  # [batch_size, num_heads]
            o_t = o_gate[:, t]  # [batch_size, num_heads]
            
            # Update matrix memory
            kv_t = torch.einsum('bhd,bhe->bhde', k_t, v_t)
            hidden_state = f_t.unsqueeze(-1).unsqueeze(-1) * hidden_state + i_t.unsqueeze(-1).unsqueeze(-1) * kv_t
            
            # Compute output
            h_t = torch.einsum('bhd,bhde->bhe', q_t, hidden_state)
            h_t = o_t.unsqueeze(-1) * h_t
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, -1)
        
        return residual + self.out_proj(output), hidden_state


class sLSTMBlock(nn.Module):
    def __init__(self, config: xLSTMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        self.conv = CausalConv1d(config.d_model, config.d_model, config.causal_conv_kernel, bias=config.use_bias)
        
        proj_dim = int(config.d_model * config.slstm_proj_factor)
        self.input_proj = nn.Linear(config.d_model, proj_dim, bias=config.use_bias)
        self.forget_proj = nn.Linear(config.d_model, proj_dim, bias=config.use_bias)
        self.output_proj = nn.Linear(config.d_model, proj_dim, bias=config.use_bias)
        self.cell_proj = nn.Linear(config.d_model, proj_dim, bias=config.use_bias)
        
        self.out_proj = nn.Linear(proj_dim, config.d_model, bias=config.use_bias)
        
        if config.norm_type == "rmsnorm":
            self.layer_norm = RMSNorm(config.d_model, config.norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
    
    def soft_cap(self, x: torch.Tensor, cap_value: float) -> torch.Tensor:
        """

        :param x:
        :param cap_value:
        :return:
        """
        return cap_value * torch.tanh(x / cap_value)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x:
        :param hidden_state:
        :return:
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        x = self.layer_norm(x)
        
        # Causal convolution
        x = self.conv(x)
        
        # Gates and projections
        i_gate = torch.sigmoid(self.soft_cap(self.input_proj(x), self.config.gate_soft_cap))
        f_gate = torch.sigmoid(self.soft_cap(self.forget_proj(x), self.config.gate_soft_cap))
        o_gate = torch.sigmoid(self.soft_cap(self.output_proj(x), self.config.gate_soft_cap))
        c_input = self.cell_proj(x)
        
        proj_dim = i_gate.shape[-1]
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, proj_dim, dtype=x.dtype, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            i_t = i_gate[:, t]
            f_t = f_gate[:, t]
            o_t = o_gate[:, t]
            
            # Update scalar memory - use the projected cell input
            c_t = c_input[:, t]
            gated_input = i_t * torch.tanh(c_t)
            hidden_state = f_t * hidden_state + gated_input
            
            # Output
            h_t = o_t * torch.tanh(hidden_state)
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)
        
        return residual + self.out_proj(output), hidden_state


class xLSTMModel(nn.Module):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Layers
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            if i < len(config.signature) and config.signature[i] == 1:
                self.blocks.append(sLSTMBlock(config, i))
            else:
                self.blocks.append(mLSTMBlock(config, i))
        
        # Output head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=config.use_bias)
        
        # Dropout
        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def init_hidden(self, batch_size: int):
        """

        :param batch_size:
        :return:
        """
        hidden_states = []
        for block in self.blocks:
            if isinstance(block, mLSTMBlock):
                hidden_states.append(torch.zeros(
                    batch_size, block.num_heads, block.head_dim, block.head_dim,
                    dtype=torch.float32, device=next(self.parameters()).device
                ))
            else:  # sLSTMBlock
                proj_dim = int(self.config.d_model * self.config.slstm_proj_factor)
                hidden_states.append(torch.zeros(
                    batch_size, proj_dim,
                    dtype=torch.float32, device=next(self.parameters()).device
                ))
        return hidden_states
    
    def forward(self, tokens: torch.Tensor, hidden_states: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """

        :param tokens:
        :param hidden_states:
        :return:
        """
        x = self.embedding(tokens)
        
        if self.dropout and self.training:
            x = self.dropout(x)
        
        if hidden_states is None:
            hidden_states = self.init_hidden(tokens.shape[0])
        
        for i, block in enumerate(self.blocks):
            x, hidden_states[i] = block(x, hidden_states[i])
            if self.dropout and self.training and i < len(self.blocks) - 1:
                x = self.dropout(x)
        
        logits = self.head(x)
        
        # Apply soft capping to output logits
        if self.config.output_logit_soft_cap > 0:
            logits = self.config.output_logit_soft_cap * torch.tanh(logits / self.config.output_logit_soft_cap)
        
        return logits, hidden_states


class MetalOptimizedxLSTM(xLSTMModel):
    def __init__(self, config: xLSTMConfig):
        super().__init__(config)
        
        # Enable MPS optimization if available
        if torch.backends.mps.is_available():
            self.device_type = "mps"
            print("Metal Performance Shaders backend enabled")
        elif torch.cuda.is_available():
            self.device_type = "cuda"
            print("CUDA backend enabled")
        else:
            self.device_type = "cpu"
            print("CPU backend enabled")
    
    def forward(self, tokens: torch.Tensor, hidden_states: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """

        :param tokens:
        :param hidden_states:
        :return:
        """
        if hasattr(self, 'device_type') and self.device_type in ["mps", "cuda"]:
            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                return super().forward(tokens, hidden_states)
        else:
            return super().forward(tokens, hidden_states)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """

        :param input_ids:
        :param max_new_tokens:
        :param temperature:
        :param top_k:
        :param top_p:
        :param do_sample:
        :param eos_token_id:
        :return:
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Prefill
        logits, hidden_states = self.forward(input_ids)
        generated = input_ids
        
        # Generation loop
        for _ in range(max_new_tokens):
            last_token = generated[:, -1:].contiguous()
            logits, hidden_states = self.forward(last_token, hidden_states)
            logits = logits[:, -1, :]  # Take last token logits
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break
        
        return generated


def create_xlstm_model(
    vocab_size: int = 50257,
    num_layers: int = 6,
    d_model: int = 512,
    signature: Tuple[int, ...] = (1, 1),
    head_dim: int = 32,
    head_num: int = 4,
    dropout: float = 0.1,
    use_metal_optimization: bool = True,
    device: str = "auto"
) -> Union[xLSTMModel, MetalOptimizedxLSTM]:
    """

    :param vocab_size:
    :param num_layers:
    :param d_model:
    :param signature:
    :param head_dim:
    :param head_num:
    :param dropout:
    :param use_metal_optimization:
    :param device:
    :return:
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    config = xLSTMConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        signature=signature,
        head_dim=head_dim,
        head_num=head_num,
        dropout=dropout,
        device=device
    )
    
    if use_metal_optimization and device in ["mps", "cuda"]:
        model = MetalOptimizedxLSTM(config)
    else:
        model = xLSTMModel(config)
    
    return model.to(device)


# Example usage
if __name__ == "__main__":
    import time
    
    print("Creating unified Metal-optimized xLSTM...")
    
    # Create model with Metal optimization
    model = create_xlstm_model(vocab_size=1000, num_layers=4, d_model=256, signature=(1, 0, 1, 0), head_num=8,
                               dropout=0.0)
    
    model.eval()
    
    # Test generation
    batch_size = 1
    seq_len = 32
    prompt = torch.randint(0, 1000, (batch_size, seq_len))
    
    if hasattr(model, 'device_type') and model.device_type != "cpu":
        prompt = prompt.to(model.device_type)
        model = model.to(model.device_type)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Prompt shape: {prompt.shape}")
    
    # Benchmark generation
    start_time = time.time()
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
    
    gen_time = time.time() - start_time
    tokens_generated = generated.shape[1] - prompt.shape[1]
    
    print(f"Generated {tokens_generated} tokens in {gen_time:.3f}s")
    print(f"Speed: {tokens_generated / gen_time:.1f} tokens/sec")
    print(f"Final sequence length: {generated.shape[1]}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("Unified xLSTM with Metal optimization complete!")