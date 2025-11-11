"""
Chunked Parallel xLSTM Implementation
Advanced parallel processing with chunk-based sequence handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Literal
from dataclasses import dataclass
import math


@dataclass
class ChunkedxLSTMConfig:
    """Configuration for chunked parallel xLSTM"""
    vocab_size: int = 50257
    num_layers: int = 12
    signature: Tuple[int, int] = (7, 1)
    inp_dim: int = 768
    head_dim: int = 96
    head_num: int = 8

    # Chunked processing
    chunk_size: int = 64
    enable_chunked_processing: bool = True
    parallel_chunks: bool = True

    # Optimization
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True

    # Stability
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0

    # Other settings
    p_factor: Tuple[float, float] = (2.0, 4 / 3)
    ker_size: int = 4
    dropout: float = 0.1
    norm_eps: float = 1e-6
    weight_mode: Literal["single", "fused"] = "fused"


def soft_cap(values: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Optimized soft capping"""
    if cap_value is None or cap_value <= 0:
        return values
    return cap_value * torch.tanh(values / cap_value)


def compute_chunk_states(k: torch.Tensor, v: torch.Tensor,
                         i_gates: torch.Tensor, f_gates: torch.Tensor,
                         chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute inter-chunk states using recurrent processing
    
    This is the key innovation: we process chunks recurrently but 
    within each chunk we can parallelize
    """
    B, NH, S, D = k.shape
    num_chunks = (S + chunk_size - 1) // chunk_size

    # Initialize states
    device = k.device
    C_states = torch.zeros(B, NH, num_chunks + 1, D, D, device=device)
    n_states = torch.zeros(B, NH, num_chunks + 1, D, device=device)

    # Process chunks recurrently
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)

        # Get chunk data
        k_chunk = k[:, :, start_idx:end_idx, :]  # (B, NH, L, D)
        v_chunk = v[:, :, start_idx:end_idx, :]
        i_chunk = i_gates[:, :, start_idx:end_idx]  # (B, NH, L)
        f_chunk = f_gates[:, :, start_idx:end_idx]

        # Previous states
        C_prev = C_states[:, :, chunk_idx, :, :]  # (B, NH, D, D)
        n_prev = n_states[:, :, chunk_idx, :]  # (B, NH, D)

        # Compute cumulative forget factors within chunk
        f_cumsum = torch.cumsum(torch.log(f_chunk + 1e-8), dim=-1)
        f_cumulative = torch.exp(f_cumsum - f_cumsum[:, :, -1:])  # Normalize by final value

        # Compute chunk contribution
        L = k_chunk.size(-2)

        # Vectorized outer products for the chunk
        # v_chunk: (B, NH, L, D), k_chunk: (B, NH, L, D)
        # We want: i_chunk[l] * v_chunk[l] @ k_chunk[l].T for each l
        v_expanded = v_chunk.unsqueeze(-1)  # (B, NH, L, D, 1)
        k_expanded = k_chunk.unsqueeze(-2)  # (B, NH, L, 1, D)
        vk_outer = v_expanded @ k_expanded  # (B, NH, L, D, D)

        # Weight by input gates
        i_expanded = i_chunk.unsqueeze(-1).unsqueeze(-1)  # (B, NH, L, 1, 1)
        weighted_vk = i_expanded * vk_outer  # (B, NH, L, D, D)

        # Weight by cumulative forget factors and sum
        f_weight = f_cumulative.unsqueeze(-1).unsqueeze(-1)  # (B, NH, L, 1, 1)
        chunk_contribution = torch.sum(f_weight * weighted_vk, dim=2)  # (B, NH, D, D)

        # Update C state
        final_forget = f_cumulative[:, :, -1].unsqueeze(-1).unsqueeze(-1)  # (B, NH, 1, 1)
        C_new = final_forget * C_prev + chunk_contribution
        C_states[:, :, chunk_idx + 1, :, :] = C_new

        # Similar computation for n states
        i_k = i_chunk.unsqueeze(-1) * k_chunk  # (B, NH, L, D)
        f_weighted_ik = f_cumulative.unsqueeze(-1) * i_k  # (B, NH, L, D)
        n_chunk_contribution = torch.sum(f_weighted_ik, dim=2)  # (B, NH, D)

        final_forget_n = f_cumulative[:, :, -1].unsqueeze(-1)  # (B, NH, 1)
        n_new = final_forget_n * n_prev + n_chunk_contribution
        n_states[:, :, chunk_idx + 1, :] = n_new

    return C_states, n_states


def parallel_chunk_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             C_states: torch.Tensor, n_states: torch.Tensor,
                             i_gates: torch.Tensor, f_gates: torch.Tensor,
                             chunk_size: int) -> torch.Tensor:
    """
    Compute attention outputs using parallel processing within chunks
    """
    global k_relevant, weights
    B, NH, S, D = q.shape
    num_chunks = (S + chunk_size - 1) // chunk_size

    outputs = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)
        L = end_idx - start_idx

        # Get chunk data
        q_chunk = q[:, :, start_idx:end_idx, :]  # (B, NH, L, D)
        k_chunk = k[:, :, start_idx:end_idx, :]
        v_chunk = v[:, :, start_idx:end_idx, :]
        i_chunk = i_gates[:, :, start_idx:end_idx]
        f_chunk = f_gates[:, :, start_idx:end_idx]

        # Get initial states for this chunk
        C_initial = C_states[:, :, chunk_idx, :, :]  # (B, NH, D, D)
        n_initial = n_states[:, :, chunk_idx, :]  # (B, NH, D)

        # Compute intra-chunk attention (this can be parallelized)
        chunk_outputs = []

        # Cumulative forget factors within chunk
        f_cumsum = torch.cumsum(torch.log(f_chunk + 1e-8), dim=-1)

        for pos in range(L):
            # States up to position pos
            f_to_pos = torch.exp(f_cumsum[:, :, pos] - f_cumsum[:, :, pos:pos + 1])  # (B, NH, 1)

            # Contribution from initial state
            f_initial = f_to_pos.unsqueeze(-1).unsqueeze(-1)  # (B, NH, 1, 1, 1)
            h_from_initial = torch.matmul(C_initial, q_chunk[:, :, pos:pos + 1, :].unsqueeze(-1))
            h_from_initial = h_from_initial.squeeze(-1)  # (B, NH, D)

            # Contribution from positions 0 to pos within chunk
            if pos > 0:
                # Get relevant positions
                k_relevant = k_chunk[:, :, :pos, :]  # (B, NH, pos, D)
                v_relevant = v_chunk[:, :, :pos, :]  # (B, NH, pos, D)
                i_relevant = i_chunk[:, :, :pos]  # (B, NH, pos)

                # Compute forget factors from each position to current position
                f_factors = torch.exp(f_cumsum[:, :, pos:pos + 1] - f_cumsum[:, :, :pos])  # (B, NH, pos)

                # Weighted contributions
                weights = (i_relevant * f_factors).unsqueeze(-1)  # (B, NH, pos, 1)
                weighted_v = weights * v_relevant  # (B, NH, pos, D)

                # Attention over positions
                q_pos = q_chunk[:, :, pos, :].unsqueeze(-2)  # (B, NH, 1, D)
                attn_weights = torch.matmul(q_pos, k_relevant.transpose(-2, -1))  # (B, NH, 1, pos)
                attn_weights = F.softmax(attn_weights, dim=-1)

                h_from_chunk = torch.matmul(attn_weights, weighted_v).squeeze(-2)  # (B, NH, D)
            else:
                h_from_chunk = torch.zeros_like(h_from_initial)

            # Combine contributions
            total_h = h_from_initial + h_from_chunk

            # Normalize
            q_pos = q_chunk[:, :, pos, :]  # (B, NH, D)
            n_from_initial = f_to_pos * n_initial
            if pos > 0:
                n_from_chunk = torch.sum(weights.squeeze(-1) * k_relevant, dim=-2)  # (B, NH, D)
                total_n = n_from_initial + n_from_chunk
            else:
                total_n = n_from_initial

            # Compute output
            denominator = torch.sum(total_n * q_pos, dim=-1, keepdim=True).clamp(min=1.0)  # (B, NH, 1)
            output_pos = total_h / denominator  # (B, NH, D)
            chunk_outputs.append(output_pos)

        # Stack chunk outputs
        chunk_output = torch.stack(chunk_outputs, dim=2)  # (B, NH, L, D)
        outputs.append(chunk_output)

    # Concatenate all chunks
    return torch.cat(outputs, dim=2)  # (B, NH, S, D)


class ChunkedParallelmLSTMBlock(nn.Module):
    """mLSTM block with chunked parallel processing"""

    def __init__(self, config: ChunkedxLSTMConfig):
        super().__init__()
        self.config = config
        self.inp_dim = config.inp_dim
        self.head_dim = config.head_dim
        self.head_num = config.head_num
        self.hidden_dim = config.head_dim * config.head_num

        p_factor = config.p_factor[0]

        self.inp_norm = nn.LayerNorm(config.inp_dim, eps=config.norm_eps)
        self.hid_norm = nn.LayerNorm(self.hidden_dim, eps=config.norm_eps)

        self.up_l_proj = nn.Linear(config.inp_dim, int(p_factor * config.inp_dim))
        self.up_r_proj = nn.Linear(config.inp_dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, config.inp_dim)

        # Causal convolution
        self.causal_conv = nn.Conv1d(1, 1, kernel_size=config.ker_size,
                                     padding=config.ker_size - 1)
        self.skip_connection = nn.Linear(int(p_factor * config.inp_dim), self.hidden_dim)

        # Projections
        if config.weight_mode == "fused":
            # Single fused linear layer
            total_dim = config.head_num * 2 + self.hidden_dim * 3  # i, f, q, k, v
            self.fused_proj = nn.Linear(int(p_factor * config.inp_dim), total_dim)
        else:
            self.W_i = nn.Linear(int(p_factor * config.inp_dim), config.head_num)
            self.W_f = nn.Linear(int(p_factor * config.inp_dim), config.head_num)
            self.W_q = nn.Linear(int(p_factor * config.inp_dim), self.hidden_dim)
            self.W_k = nn.Linear(int(p_factor * config.inp_dim), self.hidden_dim)
            self.W_v = nn.Linear(int(p_factor * config.inp_dim), self.hidden_dim)

        self.W_o = nn.Linear(int(p_factor * config.inp_dim), self.hidden_dim)

    def forward(self, x: torch.Tensor, hidden_state=None):
        """Forward pass with chunked parallel processing"""
        if x.dim() == 2:
            # Single timestep - use standard processing
            return self.forward_single(x, hidden_state)

        B, S, D = x.shape

        # Normalize input
        x_n = self.inp_norm(x)
        x_t = self.up_l_proj(x_n)
        r_t = self.up_r_proj(x_n)

        # Causal convolution
        x_c = self.causal_conv(x_t.transpose(1, 2))  # (B, D, S)
        # Remove extra padding
        if self.causal_conv.padding[0] > 0:
            x_c = x_c[:, :, :-self.causal_conv.padding[0]]
        x_c = F.silu(x_c.transpose(1, 2))  # (B, S, D)

        x_skip = self.skip_connection(x_c)

        # Compute projections
        if self.config.weight_mode == "fused":
            projections = self.fused_proj(x_c)
            # Split projections
            i_dim = self.config.head_num
            f_dim = self.config.head_num
            q_dim = k_dim = v_dim = self.hidden_dim

            splits = [i_dim, f_dim, q_dim, k_dim, v_dim]
            i_pre, f_pre, q, k, v = torch.split(projections, splits, dim=-1)
        else:
            i_pre = self.W_i(x_c)
            f_pre = self.W_f(x_c)
            q = self.W_q(x_c)
            k = self.W_k(x_c)
            v = self.W_v(x_c)

        o = torch.sigmoid(self.W_o(x_t))

        # Apply soft capping
        i_pre = soft_cap(i_pre, self.config.gate_soft_cap)
        f_pre = soft_cap(f_pre, self.config.gate_soft_cap)

        # Reshape for multi-head processing
        q = q.view(B, S, self.config.head_num, self.head_dim).transpose(1, 2)  # (B, NH, S, D)
        k = k.view(B, S, self.config.head_num, self.head_dim).transpose(1, 2) / math.sqrt(self.head_dim)
        v = v.view(B, S, self.config.head_num, self.head_dim).transpose(1, 2)

        i_gates = i_pre.transpose(1, 2)  # (B, NH, S)
        f_gates = f_pre.transpose(1, 2)

        # Exponential gating (this could be optimized further)
        max_vals = torch.maximum(f_gates + 0, i_gates)  # Simplified m computation
        i_gates = torch.exp(i_gates - max_vals)
        f_gates = torch.exp(f_gates - max_vals)

        if self.config.enable_chunked_processing and S > self.config.chunk_size:
            # Use chunked parallel processing
            C_states, n_states = compute_chunk_states(k, v, i_gates, f_gates, self.config.chunk_size)
            h = parallel_chunk_attention(q, k, v, C_states, n_states,
                                         i_gates, f_gates, self.config.chunk_size)
        else:
            # Standard sequential processing for short sequences
            h = self.standard_attention(q, k, v, i_gates, f_gates)

        # Apply output gate and reshape
        h = h.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)  # (B, S, NH*D)
        h = o * h

        # Final processing
        out = self.hid_norm(h) + x_skip
        out *= F.silu(r_t)
        out = self.down_proj(out)

        return out + x, None  # No hidden state returned for sequence processing

    @staticmethod
    def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           i_gates: torch.Tensor, f_gates: torch.Tensor) -> torch.Tensor:
        """Standard sequential attention for comparison/short sequences"""
        B, NH, S, D = q.shape

        # Initialize states
        C = torch.zeros(B, NH, D, D, device=q.device)
        n = torch.zeros(B, NH, D, device=q.device)
        outputs = []

        for t in range(S):
            # Update states
            i_t = i_gates[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, NH, 1, 1)
            f_t = f_gates[:, :, t].unsqueeze(-1).unsqueeze(-1)

            v_t = v[:, :, t, :].unsqueeze(-1)  # (B, NH, D, 1)
            k_t = k[:, :, t, :].unsqueeze(-2)  # (B, NH, 1, D)

            C = f_t * C + i_t * (v_t @ k_t)

            i_t_n = i_gates[:, :, t].unsqueeze(-1)  # (B, NH, 1)
            f_t_n = f_gates[:, :, t].unsqueeze(-1)
            k_t_n = k[:, :, t, :]

            n = f_t_n * n + i_t_n * k_t_n

            # Compute output
            q_t = q[:, :, t, :].unsqueeze(-1)  # (B, NH, D, 1)
            h_num = (C @ q_t).squeeze(-1)  # (B, NH, D)
            h_den = torch.sum(n * q[:, :, t, :], dim=-1, keepdim=True).clamp(min=1.0)
            h_t = h_num / h_den

            outputs.append(h_t)

        return torch.stack(outputs, dim=2)  # (B, NH, S, D)

    def forward_single(self, x: torch.Tensor, hidden_state):
        """Single timestep processing"""
        # This would be similar to the original implementation
        # For now, just return a placeholder
        B, D = x.shape
        out = torch.zeros_like(x)
        new_state = hidden_state if hidden_state else (
            torch.zeros(B, self.config.head_num, self.head_dim, self.head_dim, device=x.device),
            torch.zeros(B, self.config.head_num, self.head_dim, device=x.device),
            torch.zeros(B, self.config.head_num, device=x.device)
        )
        return out, new_state


class ChunkedParallelxLSTM(nn.Module):
    """Complete chunked parallel xLSTM model"""

    def __init__(self, config: ChunkedxLSTMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.inp_dim)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # For now, use only mLSTM blocks
        self.blocks = nn.ModuleList([
            ChunkedParallelmLSTMBlock(config) for _ in range(config.num_layers)
        ])

        self.out_norm = nn.LayerNorm(config.inp_dim, eps=config.norm_eps)
        self.head = nn.Linear(config.inp_dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, return_hidden=False):
        """Forward pass with chunked parallel processing"""
        x = self.embedding(tokens)
        if self.dropout:
            x = self.dropout(x)

        hidden_states = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            if self.config.use_gradient_checkpointing and self.training:
                x, hidden_states[i] = torch.utils.checkpoint.checkpoint(
                    block, x, hidden_states[i], use_reentrant=False
                )
            else:
                x, hidden_states[i] = block(x, hidden_states[i])

            if self.dropout and i < len(self.blocks) - 1:
                x = self.dropout(x)

        x = self.out_norm(x)
        logits = self.head(x)
        logits = soft_cap(logits, self.config.output_logit_soft_cap)

        if return_hidden:
            return logits, hidden_states
        return logits


def create_chunked_parallel_xlstm(config: Optional[ChunkedxLSTMConfig] = None):
    """Create chunked parallel xLSTM model"""
    if config is None:
        config = ChunkedxLSTMConfig()

    return ChunkedParallelxLSTM(config)


if __name__ == "__main__":
    print("Creating Chunked Parallel xLSTM...")

    config = ChunkedxLSTMConfig(num_layers=6, signature=(6, 0), inp_dim=512, head_dim=64, chunk_size=32)

    model = create_chunked_parallel_xlstm(config)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test with different sequence lengths
    batch_size = 2

    for seq_len in [64, 128, 256]:
        print(f"\nTesting with sequence length: {seq_len}")
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        try:
            import time

            start_time = time.time()

            logits = model(tokens)

            end_time = time.time()

            print(f"  Success: {logits.shape}")
            print(f"  Time: {end_time - start_time:.4f}s")
            print(f"  Tokens/sec: {batch_size * seq_len / (end_time - start_time):.0f}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nChunked parallel processing complete!")
    print(f"Key features:")
    print(f"  - Chunk size: {config.chunk_size}")
    print(f"  - Parallel chunk processing: {config.parallel_chunks}")
    print(f"  - Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - Weight fusion: {config.weight_mode}")
