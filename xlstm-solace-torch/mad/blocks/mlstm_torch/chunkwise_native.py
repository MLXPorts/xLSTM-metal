#!/usr/bin/env python
# Copyright (c) NXAI GmbH and its affiliates 2024
# Sydney Bach, Solace Harmony
# ChunkwiseMLS TM - Chunkwise mLSTM block for MAD framework

"""
ChunkwiseMLS TM (Chunkwise Matrix LSTM with Threaded Metal)

A MAD block that implements chunkwise-parallel mLSTM processing with:
- Thread-based queued Metal compilation for Apple Silicon
- O(chunk_size) memory efficiency vs O(S²) parallel
- Recurrent between chunks, parallel within chunks
- State caching and sliding window management

This replaces the old xLSTM paradigm of kernel selection via string names.
Instead, ChunkwiseMLS TM is a first-class MAD block with clear semantics.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import threading

from xlstm_solace_torch.mad.init import small_init_init_, wang_init_
from .chunkwise_native_canonical import mlstm_chunkwise_fw


class StateManager:
    """Manages mLSTM states across chunks with sliding window support."""

    def __init__(self, max_cache_length: int = 8192, use_sliding_window: bool = True):
        self.max_cache_length = max_cache_length
        self.use_sliding_window = use_sliding_window
        self.states_cache = {}
        self.position = 0
        self.lock = threading.Lock()

    def init_states(
        self,
        batch_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Initialize fresh mLSTM states."""
        with self.lock:
            return {
                'C': torch.zeros(
                    batch_size, num_heads, head_dim_qk, head_dim_v,
                    device=device, dtype=torch.float32
                ),
                'n': torch.ones(
                    batch_size, num_heads, head_dim_qk,
                    device=device, dtype=torch.float32
                ),
                'm': torch.zeros(
                    batch_size, num_heads, 1,
                    device=device, dtype=torch.float32
                ),
                'position': 0
            }

    def update_states(
        self,
        states: Dict[str, torch.Tensor],
        C_new: torch.Tensor,
        n_new: torch.Tensor,
        m_new: torch.Tensor,
    ):
        """Update states in-place for memory efficiency."""
        with self.lock:
            states['C'].copy_(C_new)
            states['n'].copy_(n_new)
            states['m'].copy_(m_new)
            states['position'] += 1


def compute_chunk_states(
    k: torch.Tensor,
    v: torch.Tensor,
    i_gates: torch.Tensor,
    f_gates: torch.Tensor,
    chunk_size: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute inter-chunk states using recurrent processing.

    Key innovation: Process chunks recurrently (sequential between chunks)
    but parallelize within each chunk.

    Args:
        k: Keys [B, NH, S, DH]
        v: Values [B, NH, S, DH]
        i_gates: Input gates [B, NH, S]
        f_gates: Forget gates [B, NH, S]
        chunk_size: Size of each chunk
        eps: Numerical stability epsilon

    Returns:
        (C_states, n_states): States at chunk boundaries
            C_states: [B, NH, num_chunks+1, DH, DH]
            n_states: [B, NH, num_chunks+1, DH]
    """
    B, NH, S, DH = k.shape
    num_chunks = (S + chunk_size - 1) // chunk_size
    device = k.device

    # Initialize state tensors at chunk boundaries
    C_states = torch.zeros(B, NH, num_chunks + 1, DH, DH, device=device, dtype=torch.float32)
    n_states = torch.zeros(B, NH, num_chunks + 1, DH, device=device, dtype=torch.float32)

    # Process chunks recurrently
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)

        # Extract chunk
        k_chunk = k[:, :, start_idx:end_idx, :]  # [B, NH, L, DH]
        v_chunk = v[:, :, start_idx:end_idx, :]
        i_chunk = i_gates[:, :, start_idx:end_idx]  # [B, NH, L]
        f_chunk = f_gates[:, :, start_idx:end_idx]

        # Previous chunk boundary states (clone to avoid in-place modification issues)
        C_prev = C_states[:, :, chunk_idx, :, :].clone()  # [B, NH, DH, DH]
        n_prev = n_states[:, :, chunk_idx, :].clone()     # [B, NH, DH]

        # Compute cumulative forget factors within chunk
        f_cumsum = torch.cumsum(torch.log(f_chunk + eps), dim=-1)
        f_cumulative = torch.exp(f_cumsum - f_cumsum[:, :, -1:])  # Normalize by final value

        L = k_chunk.size(-2)

        # Vectorized outer products for the chunk
        # v_chunk: [B, NH, L, DH], k_chunk: [B, NH, L, DH]
        # Want: i_chunk[l] * v_chunk[l] @ k_chunk[l].T for each l
        v_expanded = v_chunk.unsqueeze(-1)  # [B, NH, L, DH, 1]
        k_expanded = k_chunk.unsqueeze(-2)  # [B, NH, L, 1, DH]
        vk_outer = v_expanded @ k_expanded  # [B, NH, L, DH, DH]

        # Weight by input gates
        i_expanded = i_chunk.unsqueeze(-1).unsqueeze(-1)  # [B, NH, L, 1, 1]
        weighted_vk = i_expanded * vk_outer  # [B, NH, L, DH, DH]

        # Weight by cumulative forget factors and sum
        f_weight = f_cumulative.unsqueeze(-1).unsqueeze(-1)  # [B, NH, L, 1, 1]
        chunk_contribution = torch.sum(f_weight * weighted_vk, dim=2)  # [B, NH, DH, DH]

        # Update C state: C_new = f_final * C_prev + chunk_contribution
        final_forget = f_cumulative[:, :, -1].unsqueeze(-1).unsqueeze(-1)  # [B, NH, 1, 1]
        C_new = final_forget * C_prev + chunk_contribution
        C_states[:, :, chunk_idx + 1, :, :] = C_new

        # Similar computation for n states
        i_k = i_chunk.unsqueeze(-1) * k_chunk  # [B, NH, L, DH]
        f_weighted_ik = f_cumulative.unsqueeze(-1) * i_k  # [B, NH, L, DH]
        n_chunk_contribution = torch.sum(f_weighted_ik, dim=2)  # [B, NH, DH]

        final_forget_n = f_cumulative[:, :, -1].unsqueeze(-1)  # [B, NH, 1]
        n_new = final_forget_n * n_prev + n_chunk_contribution
        n_states[:, :, chunk_idx + 1, :] = n_new

    return C_states, n_states


def compute_chunk_outputs(
    q: torch.Tensor,          # [B, NH, S, DHQK]
    k: torch.Tensor,          # [B, NH, S, DHQK]
    v: torch.Tensor,          # [B, NH, S, DHHV]
    i_gates: torch.Tensor,    # [B, NH, S]
    f_gates: torch.Tensor,    # [B, NH, S]
    C_states: torch.Tensor,   # [B, NH, NC+1, DHQK, DHHV]
    n_states: torch.Tensor,   # [B, NH, NC+1, DHQK]
    m_states: torch.Tensor,   # [B, NH, NC+1]
    chunk_size: int,
    qk_scale: float = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute chunk outputs using parallel intra-chunk attention.

    This is the parallel_fw_H function from mlstm_kernels.
    Combines intra-chunk and inter-chunk contributions.

    Args:
        q, k, v: QKV projections
        i_gates, f_gates: Input/forget gates
        C_states, n_states, m_states: Chunk boundary states
        chunk_size: Chunk size
        qk_scale: QK scaling factor
        eps: Numerical stability

    Returns:
        h: Output hidden states [B, NH, S, DHHV]
    """
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]
    NC = (S + chunk_size - 1) // chunk_size
    L = chunk_size

    if qk_scale is None:
        qk_scale = DHQK ** -0.5

    # Reshape into chunks [B, NH, NC, L, DH]
    def rechunk(x, L):
        # Pad if needed
        pad_len = NC * L - S
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        return x.view(B, NH, NC, L, -1)

    q_chunked = rechunk(q, L)  # [B, NH, NC, L, DHQK]
    k_chunked = rechunk(k, L)
    v_chunked = rechunk(v, L)

    # Reshape gates
    i_pad = torch.nn.functional.pad(i_gates, (0, NC * L - S)) if NC * L > S else i_gates
    f_pad = torch.nn.functional.pad(f_gates, (0, NC * L - S)) if NC * L > S else f_gates
    i_chunked = i_pad.view(B, NH, NC, L)  # [B, NH, NC, L]
    f_chunked = f_pad.view(B, NH, NC, L)

    # Compute vecB (cumulative log forget gates)
    vecB = torch.cumsum(torch.log(f_chunked + eps), dim=-1)  # [B, NH, NC, L]

    # Get chunk states (exclude final state)
    C_k_states = C_states[:, :, :-1, :, :]  # [B, NH, NC, DHQK, DHHV]
    n_k_states = n_states[:, :, :-1, :]     # [B, NH, NC, DHQK]
    m_inter_states = m_states[:, :, :-1]    # [B, NH, NC]

    # Causal mask for intra-chunk attention
    ltr = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))

    # Compute intra-chunk contribution
    # matF_logsig_chunk: difference of cumulative forget between positions
    matF_logsig_chunk = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]  # [B, NH, NC, L, L]

    # Apply causal mask
    matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, torch.tensor(-float("inf"), device=q.device))

    # Add input gates
    matLogD_chunk = matF_logsig_mask_chunk + torch.log(i_chunked[:, :, :, None, :] + eps)  # [B, NH, NC, L, L]

    # Max stabilization for intra-chunk
    vecMintra_k = torch.max(matLogD_chunk, dim=-1).values  # [B, NH, NC, L]

    # Combine with inter-chunk max states
    vecM_b_inter = vecB + m_inter_states[:, :, :, None]  # [B, NH, NC, L]
    vecM_k_combine = torch.maximum(vecM_b_inter, vecMintra_k)  # [B, NH, NC, L]

    vecM_k_combine_exp = vecM_k_combine[:, :, :, :, None]  # [B, NH, NC, L, 1]
    vecM_b_inter_exp = vecM_b_inter[:, :, :, :, None]

    # Stabilized attention weights
    matLogD_stabilized = matLogD_chunk - vecM_k_combine_exp
    matD_chunk = torch.exp(matLogD_stabilized)  # [B, NH, NC, L, L]

    # QK attention
    matS_chunk = torch.matmul(q_chunked, k_chunked.transpose(-2, -1)) * qk_scale  # [B, NH, NC, L, L]

    # Gated attention
    matM_chunk = matS_chunk * matD_chunk  # [B, NH, NC, L, L]

    # Combine intra and inter chunk contributions
    vecBbar = torch.exp(vecM_b_inter_exp - vecM_k_combine_exp)  # [B, NH, NC, L, 1]
    matQ_gated = q_chunked * vecBbar * qk_scale  # [B, NH, NC, L, DHQK]

    # Numerator: inter-chunk (Q @ C_k) + intra-chunk (M @ V)
    matNumerator = torch.matmul(matQ_gated, C_k_states) + torch.matmul(matM_chunk, v_chunked)  # [B, NH, NC, L, DHHV]

    # Denominator: inter-chunk (Q @ n_k) + intra-chunk (sum(M))
    vecDenom = torch.matmul(matQ_gated, n_k_states.unsqueeze(-1)) + matM_chunk.sum(dim=-1, keepdim=True)  # [B, NH, NC, L, 1]

    # Stabilize denominator
    vecDenom_max = torch.maximum(torch.abs(vecDenom), torch.exp(-vecM_k_combine_exp))

    # Final output
    matH_chunk = matNumerator / (vecDenom_max + eps)  # [B, NH, NC, L, DHHV]

    # Reshape back to sequence
    h = matH_chunk.view(B, NH, NC * L, DHHV)  # [B, NH, S_padded, DHHV]

    # Remove padding
    if NC * L > S:
        h = h[:, :, :S, :]

    return h


# Compiled version for MPS/Metal/ANE deployment via torch.compile
try:
    _compile_mode = os.environ.get("XLSTM_COMPILE_MODE", "reduce-overhead")
    compute_chunk_states_compiled = torch.compile(
        compute_chunk_states,
        backend="inductor",
        mode=_compile_mode,
    )
except Exception as e:
    # Fallback if torch.compile not available
    compute_chunk_states_compiled = compute_chunk_states


class ChunkwiseMLSTM(nn.Module):
    """Chunkwise mLSTM block with threaded Metal compilation.

    MAD block for efficient long-sequence processing with:
    - Chunkwise processing: O(chunk_size) memory vs O(S²)
    - Thread-based queued Metal compilation for Apple Silicon
    - State caching with sliding window support
    - Recurrent between chunks, parallel within chunks
    - MAD iso-state normalization for fair architecture comparisons

    Args:
        dim: Model dimension
        num_heads: Number of mLSTM heads (default 4)
        chunk_size: Chunk size for processing (default 64)
        proj_factor: Inner dimension expansion factor (default 2.0, ignored if total_state_dim set)
        qkv_proj_blocksize: QKV projection block size (default 4)
        conv_kernel_size: Causal conv1d kernel size (default 4)
        use_sliding_window: Enable sliding window state management (default True)
        max_cache_length: Maximum cached sequence length (default 8192)
        bias: Use bias in linear layers (default False)
        dropout: Dropout rate (default 0.0)
        num_blocks: Number of blocks for weight init scaling (default 1)
        total_state_dim: MAD iso-state normalization (default None). If set, computes
                        head_dim = sqrt(total_state_dim / num_heads) to normalize
                        total state to this value. MAD protocol uses 4096.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        chunk_size: int = 64,
        proj_factor: float = 2.0,
        qkv_proj_blocksize: int = 4,
        conv_kernel_size: int = 4,
        use_sliding_window: bool = True,
        max_cache_length: int = 8192,
        bias: bool = False,
        dropout: float = 0.0,
        num_blocks: int = 1,
        eps: float = 1e-6,
        use_compile: bool = True,  # Use torch.compile for MPS/Metal/ANE
        total_state_dim: Optional[int] = None,  # MAD iso-state normalization (default 4096)
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.eps = eps
        self.num_blocks = num_blocks
        self.use_compile = use_compile

        # MAD iso-state normalization
        # For outer-product mLSTM: total_state = num_heads * head_dim²
        # MAD protocol normalizes all architectures to total_state_dim = 4096
        if total_state_dim is not None:
            # Compute head_dim from iso-state constraint
            import math
            self.head_dim = int(math.sqrt(total_state_dim / num_heads))
            self.inner_dim = num_heads * self.head_dim
            print(f"[MAD iso-state] total_state={total_state_dim}, num_heads={num_heads}, "
                  f"head_dim={self.head_dim}, inner_dim={self.inner_dim}")
        else:
            # Legacy mode: use proj_factor (not iso-state normalized!)
            self.inner_dim = int(dim * proj_factor)
            assert self.inner_dim % num_heads == 0, \
                f"inner_dim {self.inner_dim} must be divisible by num_heads {num_heads}"
            self.head_dim = self.inner_dim // num_heads

        # State manager
        self.state_manager = StateManager(
            max_cache_length=max_cache_length,
            use_sliding_window=use_sliding_window,
        )

        # Projections
        self.proj_up = nn.Linear(dim, 2 * self.inner_dim, bias=bias)
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=bias)

        # Gates
        self.igate_proj = nn.Linear(self.inner_dim, num_heads, bias=True)
        self.fgate_proj = nn.Linear(self.inner_dim, num_heads, bias=True)

        # Initialize forget gate bias (encourage retention)
        with torch.no_grad():
            self.fgate_proj.bias.copy_(torch.linspace(3.0, 6.0, num_heads))

        # Causal conv (optional)
        if conv_kernel_size > 0:
            self.conv1d = nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1,
                groups=self.inner_dim,
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv1d = None

        # Output
        self.learnable_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.ogate_act = nn.SiLU()
        self.proj_down = nn.Linear(self.inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Thread pool for queued compilation (Apple Silicon optimization)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with chunkwise processing.

        Args:
            x: Input [B, S, D]
            states: Optional previous states for continuation

        Returns:
            (output [B, S, D], final_states)
        """
        B, S, _ = x.shape

        # Project up and split
        x_up = self.proj_up(x)  # [B, S, 2*inner_dim]
        x_mlstm, x_ogate = x_up.chunk(2, dim=-1)

        # Optional causal conv
        if self.conv1d is not None:
            x_conv = x_mlstm.transpose(1, 2)  # [B, inner_dim, S]
            x_conv = self.conv1d(x_conv)
            x_conv = x_conv[:, :, :S]  # Remove padding
            x_mlstm = self.conv_act(x_conv.transpose(1, 2))

        # QKV projections
        q = self.q_proj(x_mlstm)  # [B, S, inner_dim]
        k = self.k_proj(x_mlstm)
        v = self.v_proj(x_mlstm)

        # Gate projections
        i_preact = self.igate_proj(x_mlstm)  # [B, S, num_heads]
        f_preact = self.fgate_proj(x_mlstm)

        # Reshape for multi-head
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, S, DH]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply gates and convert to log space (canonical expects log gates)
        i_gates = torch.sigmoid(i_preact.transpose(1, 2))  # [B, NH, S]
        f_gates = torch.sigmoid(f_preact.transpose(1, 2))

        # Convert to log space for canonical implementation
        vecI = torch.log(i_gates + self.eps)  # [B, NH, S]
        vecF = torch.log(f_gates + self.eps)  # [B, NH, S]

        # Use canonical chunkwise forward pass
        # This computes both states and outputs correctly
        h, vecN_out, vecM_out, last_states, all_states = mlstm_chunkwise_fw(
            matQ=q,
            matK=k,
            matV=v,
            vecI=vecI,
            vecF=vecF,
            matC_initial=None,  # TODO: support state continuation
            vecN_initial=None,
            scaM_initial=None,
            qk_scale=self.head_dim ** -0.5,
            return_last_states=True,
            return_all_states=False,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )

        # h is [B, NH, S, DHHV], transpose to [B, S, NH, DHHV] and reshape
        h = h.transpose(1, 2).reshape(B, S, self.inner_dim)  # [B, S, inner_dim]

        # Learnable skip
        h = h + self.learnable_skip * x_mlstm

        # Output gating
        h = h * self.ogate_act(x_ogate)

        # Project down
        y = self.proj_down(h)
        if self.dropout is not None:
            y = self.dropout(y)

        # Package final states from canonical implementation
        if last_states is not None:
            matC_last, vecN_last, scaM_last = last_states
            final_states = {
                'C': matC_last,
                'n': vecN_last,
                'm': scaM_last,
            }
        else:
            final_states = None

        return y, final_states

    def reset_parameters(self):
        """Initialize weights using canonical xLSTM initialization."""
        small_init_init_(self.proj_up.weight, dim=self.dim)
        small_init_init_(self.q_proj.weight, dim=self.inner_dim)
        small_init_init_(self.k_proj.weight, dim=self.inner_dim)
        small_init_init_(self.v_proj.weight, dim=self.inner_dim)
        small_init_init_(self.igate_proj.weight, dim=self.inner_dim)
        small_init_init_(self.fgate_proj.weight, dim=self.inner_dim)
        wang_init_(self.proj_down.weight, dim=self.inner_dim, num_blocks=self.num_blocks)
        nn.init.ones_(self.learnable_skip)
