
"""
xLSTM implementation for PyTorch
Based on Beck et al. (2024) - "xLSTM: Extended Long Short-Term Memory"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Literal
from dataclasses import dataclass
import math


@dataclass
class xLSTMConfig:
    """Configuration for xLSTM model"""
    vocab_size: int = 50257
    num_layers: int = 12
    signature: Tuple[int, int] = (7, 1)  # (num_mLSTM, num_sLSTM)
    inp_dim: int = 768
    head_dim: int = 96
    head_num: int = 8
    
    # Dimension scaling factors
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    
    # Projection factors
    p_factor: Tuple[float, float] = (2.0, 4/3)  # (mLSTM_factor, sLSTM_factor)
    ker_size: int = 4
    dropout: float = 0.1
    
    # Stability features
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    
    # Normalization
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    use_bias: bool = False
    
    # Advanced features  
    weight_mode: Literal["single", "fused"] = "single"
    add_out_norm: bool = True
    
    # Feed-forward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64


def soft_cap(values: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Soft caps a tensor using tanh to prevent gradient explosion.
    
    Performs: cap_value * tanh(values / cap_value)
    
    Args:
        values: The tensor to cap
        cap_value: The soft cap value
        
    Returns:
        The soft-capped values
    """
    if cap_value is None or cap_value <= 0:
        return values
    return cap_value * torch.tanh(values / cap_value)


class CausalConv1d(nn.Module):
    """Causal 1D convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        """

        :param x:
        :return:
        """
        # x shape: (batch, channels, length)
        out = self.conv(x)
        # Remove future positions for causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class BlockLinear(nn.Module):
    """Block diagonal linear layer"""
    def __init__(self, block_specs):
        super().__init__()
        self.blocks = nn.ModuleList()
        for in_dim, out_dim in block_specs:
            self.blocks.append(nn.Linear(in_dim, out_dim))
    
    def forward(self, x):
        """

        :param x:
        :return:
        """
        # Split input and apply block-wise linear transformation
        block_size = x.shape[-1] // len(self.blocks)
        outputs = []
        for i, block in enumerate(self.blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size
            block_input = x[..., start_idx:end_idx]
            outputs.append(block(block_input))
        return torch.cat(outputs, dim=-1)


def enlarge_as(x, target):
    """Expand tensor x to match target dimensions"""
    while len(x.shape) < len(target.shape):
        x = x.unsqueeze(-1)
    return x


class sLSTMBlock(nn.Module):
    """Scalar LSTM block with exponential gating and state normalization"""
    def __init__(self, inp_dim, head_dim, head_num, p_factor=4/3, ker_size=4):
        super().__init__()
        self.inp_dim = inp_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim = head_dim * head_num
        
        self.inp_norm = nn.LayerNorm(config.inp_dim, eps=config.norm_eps)
        self.hid_norm = MultiHeadLayerNorm(config.head_num, config.head_dim, eps=config.norm_eps, 
                                          force_float32_reductions=config.norm_reduction_force_float32)
        
        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        
        self.W_z = nn.Linear(inp_dim, self.hidden_dim)
        self.W_i = nn.Linear(inp_dim, self.hidden_dim)
        self.W_o = nn.Linear(inp_dim, self.hidden_dim)
        self.W_f = nn.Linear(inp_dim, self.hidden_dim)
        
        self.R_z = BlockLinear([(head_dim, head_dim)] * head_num)
        self.R_i = BlockLinear([(head_dim, head_dim)] * head_num)
        self.R_o = BlockLinear([(head_dim, head_dim)] * head_num)
        self.R_f = BlockLinear([(head_dim, head_dim)] * head_num)
        
        proj_dim = int(p_factor * self.hidden_dim)
        self.up_proj = nn.Linear(self.hidden_dim, 2 * proj_dim)
        self.down_proj = nn.Linear(proj_dim, inp_dim)
    
    @property
    def device(self):
        """

        :return:
        """
        return next(self.parameters()).device
    
    def init_hidden(self, batch_size):
        """Initialize hidden states"""
        n_0 = torch.ones(batch_size, self.hidden_dim, device=self.device)
        c_0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h_0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        m_0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        return c_0, n_0, h_0, m_0
    
    def update_hidden_inplace(self, hidden_state, new_state):
        """Update hidden states in-place for memory efficiency"""
        c_tm1, n_tm1, h_tm1, m_tm1 = hidden_state
        c_new, n_new, h_new, m_new = new_state
        
        c_tm1.copy_(c_new)
        n_tm1.copy_(n_new)
        h_tm1.copy_(h_new)
        m_tm1.copy_(m_new)
        
        return hidden_state
    
    def forward_step(self, x, hidden_state, use_conv=False):
        """Process single time step"""
        c_tm1, n_tm1, h_tm1, m_tm1 = hidden_state
        
        x_t = self.inp_norm(x)
        
        if use_conv:
            x_c = self.causal_conv(x_t.unsqueeze(1))
            x_c = F.silu(x_c.squeeze(1))
        else:
            x_c = x_t
        
        i_t = soft_cap(self.W_i(x_c) + self.R_i(h_tm1), 15.0)
        f_t = soft_cap(self.W_f(x_c) + self.R_f(h_tm1), 15.0)
        z_t = self.W_z(x_t) + self.R_z(h_tm1)
        o_t = self.W_o(x_t) + self.R_o(h_tm1)
        
        m_t = torch.max(f_t + m_tm1, i_t)
        i_t = torch.exp(i_t - m_t)
        f_t = torch.exp(f_t - m_t + m_tm1)
        
        z_t = torch.tanh(z_t)
        o_t = torch.sigmoid(o_t)
        
        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t
        h_t = o_t * (c_t / n_t.clamp(min=1.0))
        
        out = self.hid_norm(h_t)
        out1, out2 = self.up_proj(out).chunk(2, dim=-1)
        out = out1 * F.gelu(out2)
        out = self.down_proj(out)
        
        return out + x, (c_t, n_t, h_t, m_t)
    
    def forward(self, x, hidden_state=None, use_conv=False):
        """Process full sequence at once"""
        if x.dim() == 2:  # Single timestep (B, D)
            return self.forward_step(x, hidden_state, use_conv)
        
        # Full sequence processing (B, S, D)
        batch_size, seq_len, _ = x.shape
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
            
        c_tm1, n_tm1, h_tm1, m_tm1 = hidden_state
        
        # Process sequence with in-place state updates for efficiency
        outputs = []
        current_state = (c_tm1, n_tm1, h_tm1, m_tm1)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            out, new_state = self.forward_step(x_t, current_state, use_conv)
            # Update state in-place for memory efficiency
            current_state = self.update_hidden_inplace(current_state, new_state)
            outputs.append(out)
            
        output_seq = torch.stack(outputs, dim=1)
        return output_seq, current_state


class mLSTMBlock(nn.Module):
    """Matrix LSTM block with covariance update rule"""
    def __init__(self, inp_dim, head_dim, head_num, p_factor=2, ker_size=4):
        super().__init__()
        self.inp_dim = inp_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim = head_dim * head_num
        
        self.inp_norm = nn.LayerNorm(config.inp_dim, eps=config.norm_eps)
        self.hid_norm = MultiHeadLayerNorm(config.head_num, config.head_dim, eps=config.norm_eps, 
                                          force_float32_reductions=config.norm_reduction_force_float32)
        
        self.up_l_proj = nn.Linear(inp_dim, int(p_factor * inp_dim))
        self.up_r_proj = nn.Linear(inp_dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, inp_dim)
        
        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        self.skip_connection = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        
        self.W_i = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_f = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_o = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        
        self.W_q = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        self.W_k = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        self.W_v = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
    
    @property
    def device(self):
        """

        :return:
        """
        return next(self.parameters()).device
    
    def init_hidden(self, batch_size):
        """Initialize hidden states"""
        c_0 = torch.zeros(batch_size, self.head_num, self.head_dim, self.head_dim, device=self.device)
        n_0 = torch.ones(batch_size, self.head_num, self.head_dim, device=self.device)
        m_0 = torch.zeros(batch_size, self.head_num, device=self.device)
        return c_0, n_0, m_0
    
    def update_hidden_inplace(self, hidden_state, new_state):
        """Update hidden states in-place for memory efficiency"""
        c_tm1, n_tm1, m_tm1 = hidden_state
        c_new, n_new, m_new = new_state
        
        c_tm1.copy_(c_new)
        n_tm1.copy_(n_new)
        m_tm1.copy_(m_new)
        
        return hidden_state
    
    def forward_step(self, x, hidden_state):
        """Process single time step"""
        bs = x.shape[0]
        c_tm1, n_tm1, m_tm1 = hidden_state
        
        x_n = self.inp_norm(x)
        
        x_t = self.up_l_proj(x_n)
        r_t = self.up_r_proj(x_n)
        
        x_c = self.causal_conv(x_t.unsqueeze(1))
        x_c = F.silu(x_c.squeeze(1))
        x_skip = self.skip_connection(x_c)
        
        q_t = self.W_q(x_c).view(bs, self.head_num, self.head_dim)
        k_t = self.W_k(x_c).view(bs, self.head_num, self.head_dim) / math.sqrt(self.head_dim)
        v_t = self.W_v(x_t).view(bs, self.head_num, self.head_dim)
        
        i_t = soft_cap(self.W_i(x_c), 15.0)
        f_t = soft_cap(self.W_f(x_c), 15.0)
        o_t = torch.sigmoid(self.W_o(x_t))
        
        m_t = torch.max(f_t + m_tm1, i_t)
        i_t = torch.exp(i_t - m_t)
        f_t = torch.exp(f_t - m_t + m_tm1)
        
        # Covariance update
        i_expanded = i_t.unsqueeze(-1).unsqueeze(-1)
        f_expanded = f_t.unsqueeze(-1).unsqueeze(-1)
        v_expanded = v_t.unsqueeze(-1)
        k_expanded = k_t.unsqueeze(-2)
        
        c_t = f_expanded * c_tm1 + i_expanded * (v_expanded @ k_expanded)
        
        f_n_expanded = f_t.unsqueeze(-1)
        i_n_expanded = i_t.unsqueeze(-1)
        n_t = f_n_expanded * n_tm1 + i_n_expanded * k_t
        
        # Compute output
        q_expanded = q_t.unsqueeze(-1)
        h_numerator = (c_t @ q_expanded).squeeze(-1)
        h_denominator = torch.sum(n_t * q_t, dim=-1, keepdim=True).clamp(min=1.0)
        h_t = o_t * (h_numerator / h_denominator).view(bs, self.hidden_dim)
        
        out = self.hid_norm(h_t) + x_skip
        out = out * F.silu(r_t)
        out = self.down_proj(out)
        
        return out + x, (c_t, n_t, m_t)
    
    def forward(self, x, hidden_state=None):
        """Process full sequence at once"""
        if x.dim() == 2:  # Single timestep (B, D)
            return self.forward_step(x, hidden_state)
        
        # Full sequence processing (B, S, D)
        batch_size, seq_len, _ = x.shape
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
            
        c_tm1, n_tm1, m_tm1 = hidden_state
        
        # Process sequence with in-place state updates for efficiency
        outputs = []
        current_state = (c_tm1, n_tm1, m_tm1)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            out, new_state = self.forward_step(x_t, current_state)
            # Update state in-place for memory efficiency
            current_state = self.update_hidden_inplace(current_state, new_state)
            outputs.append(out)
            
        output_seq = torch.stack(outputs, dim=1)
        return output_seq, current_state


class xLSTM(nn.Module):
    """xLSTM model combining sLSTM and mLSTM blocks"""
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.inp_dim)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        m_num, s_num = config.signature
        block_types = [True] * m_num + [False] * s_num
        
        # Cycle through block types for the specified number of layers
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            block_type = block_types[i % len(block_types)]
            if block_type:  # mLSTM block
                self.blocks.append(mLSTMBlock(config))
            else:  # sLSTM block
                self.blocks.append(sLSTMBlock(config))
        
        # Output normalization
        if config.add_out_norm:
            self.out_norm = nn.LayerNorm(config.inp_dim, eps=config.norm_eps)
        else:
            self.out_norm = nn.Identity()
            
        self.head = nn.Linear(config.inp_dim, config.vocab_size, bias=config.use_bias)
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for all blocks"""
        return [block.init_hidden(batch_size) for block in self.blocks]
    
    def forward(self, tokens, hidden_states=None, return_hidden=False):
        """
        Forward pass through xLSTM
        
        Args:
            tokens: Input token indices (batch_size, seq_len)
            hidden_states: Optional initial hidden states
            return_hidden: Whether to return final hidden states
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            hidden_states: Final hidden states (if return_hidden=True)
        """
        batch_size, seq_len = tokens.shape
        
        # Embed tokens
        x = self.embedding(tokens)
        if self.dropout:
            x = self.dropout(x)
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size)
        
        # Process sequence through blocks (sequence-level processing)
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, hidden_states[i])
            # Update hidden state in-place for memory efficiency
            if hasattr(block, 'update_hidden_inplace'):
                hidden_states[i] = block.update_hidden_inplace(hidden_states[i], new_state)
            else:
                hidden_states[i] = new_state
            if self.dropout and i < len(self.blocks) - 1:
                x = self.dropout(x)
        
        # Output normalization
        x = self.out_norm(x)
        
        logits = self.head(x)
        logits = soft_cap(logits, 30.0)  # Output logit soft capping
        
        if return_hidden:
            return logits, hidden_states
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        hidden_states: Optional[List] = None
    ) -> torch.Tensor:
        """
        Generate text using the xLSTM model
        
        Args:
            input_ids: Input token ids (batch_size, seq_len)
            generation_config: Generation configuration
            hidden_states: Optional initial hidden states
            
        Returns:
            generated_ids: Generated token ids (batch_size, max_length)
        """
        if generation_config is None:
            generation_config = GenerationConfig()
            
        batch_size, input_len = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated_ids = torch.full(
            (batch_size, generation_config.max_length),
            generation_config.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )
        generated_ids[:, :input_len] = input_ids
        
        # Process prefill
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size)
        
        # Process input sequence
        _, hidden_states = self.forward(input_ids, hidden_states, return_hidden=True)
        
        # Generation loop
        for step in range(input_len, generation_config.max_length):
            # Get logits for next token
            last_token = generated_ids[:, step-1:step]
            logits, hidden_states = self.forward(last_token, hidden_states, return_hidden=True)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply repetition penalty
            if generation_config.repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, generated_ids[:, :step], generation_config.repetition_penalty
                )
            
            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=generation_config.top_k,
                top_p=generation_config.top_p
            )
            
            # Sample next token
            next_token = sample_token(
                filtered_logits,
                temperature=generation_config.temperature,
                do_sample=generation_config.do_sample
            )
            
            generated_ids[:, step:step+1] = next_token
            
            # Check for EOS token
            if torch.all(next_token == generation_config.eos_token_id):
                break
                
        return generated_ids


def create_xlstm_model(config: Optional[xLSTMConfig] = None, device: str = 'cpu') -> xLSTM:
    """
    Create an xLSTM model with specified configuration
    
    Args:
        config: xLSTM configuration object. If None, uses default config.
        device: Device to place model on
        
    Returns:
        xLSTM model instance
    """
    if config is None:
        config = xLSTMConfig()
    
    model = xLSTM(config)
    return model.to(device)


class GenerationConfig:
    """Configuration for text generation"""
    def __init__(
        self,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 0,
        repetition_penalty: float = 1.0
    ):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply repetition penalty to logits"""
    if penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    for i in range(batch_size):
        for token_id in input_ids[i]:
            if logits[i, token_id] < 0:
                logits[i, token_id] *= penalty
            else:
                logits[i, token_id] /= penalty
    return logits


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or top-p (nucleus) sampling"""
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    
    return logits


def sample_token(logits: torch.Tensor, temperature: float = 1.0, do_sample: bool = True) -> torch.Tensor:
    """Sample next token from logits"""
    if not do_sample or temperature == 0:
        # Greedy sampling
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Apply temperature
    logits = logits / temperature
    
    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = xLSTMConfig(vocab_size=1000, num_layers=4, signature=(1, 1), inp_dim=256, head_dim=32)
    model = create_xlstm_model(config, device=device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    logits = model(tokens)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: torch.Size([{batch_size}, {seq_len}, 1000])")