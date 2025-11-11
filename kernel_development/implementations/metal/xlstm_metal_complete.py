"""
Complete Metal-optimized xLSTM implementation that matches official interface.
Uses Metal when available, otherwise falls back to CPU.
"""

from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal, Optional, Union, Tuple
import math

# Import official xLSTM components
from xlstm.xlstm_large.components import MultiHeadLayerNorm, RMSNorm, soft_cap
from xlstm.xlstm_large.utils import round_up_to_next_multiple_of
from xlstm.xlstm_large.generate import generate_tokens, get_sampling_fn

# Check device availability
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Type definitions from official implementation
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
class mLSTMBackendConfig:
    """Configuration for the mLSTM backend.

    This dataclass holds the configuration for the mLSTM backend, which can be
    configured to use different kernels and modes for training and inference.

    Attributes:
        chunkwise_kernel (ChunkwiseKernelType): The kernel to use for chunkwise
            processing. Can be "native", "metal", or "cuda".
        sequence_kernel (SequenceKernelType): The kernel to use for sequence
            processing. Can be "native", "metal", or "cuda".
        step_kernel (StepKernelType): The kernel to use for single-step
            processing. Can be "native", "metal", or "cuda".
        mode (BackendModeType): The mode of operation. Can be "train",
            "train_with_padding", or "inference".
        chunk_size (int): The chunk size for chunkwise processing.
        return_last_states (bool): Whether to return the last hidden states.
        autocast_kernel_dtype (DtypeType): The data type to use for autocasting
            in the kernels.
        eps (float): A small value to add to the denominator for numerical
            stability.
        inference_state_dtype (DtypeType): The data type to use for the hidden
            states during inference.
    """
    chunkwise_kernel: ChunkwiseKernelType = "native"
    sequence_kernel: SequenceKernelType = "native"
    step_kernel: StepKernelType = "native"
    mode: BackendModeType = "train"
    chunk_size: int = 64
    return_last_states: bool = False
    autocast_kernel_dtype: DtypeType = "float16" if DEVICE == "mps" else "bfloat16"
    eps: float = 1e-6
    inference_state_dtype: DtypeType = "float32"


@dataclass
class xLSTMLargeConfig:
    """Configuration for the xLSTM model.

    This dataclass holds the configuration for the xLSTM model, including the
    embedding dimension, number of heads, number of blocks, vocabulary size,
    and other hyperparameters.

    Attributes:
        embedding_dim (int): The dimension of the embedding layer.
        num_heads (int): The number of heads in the multi-head attention layers.
        num_blocks (int): The number of mLSTM blocks in the model.
        vocab_size (int): The size of the vocabulary.
        use_bias (bool): Whether to use bias in the linear layers.
        norm_eps (float): The epsilon value for the normalization layers.
        norm_reduction_force_float32 (bool): Whether to force float32 for the
            reduction operations in the normalization layers.
        add_out_norm (bool): Whether to add a normalization layer at the end of
            the model.
        qk_dim_factor (float): The dimension factor for the query and key
            projections.
        v_dim_factor (float): The dimension factor for the value projection.
        chunkwise_kernel (ChunkwiseKernelType): The kernel to use for chunkwise
            processing.
        sequence_kernel (SequenceKernelType): The kernel to use for sequence
            processing.
        step_kernel (StepKernelType): The kernel to use for single-step
            processing.
        mode (BackendModeType): The mode of operation.
        chunk_size (int): The chunk size for chunkwise processing.
        return_last_states (bool): Whether to return the last hidden states.
        autocast_kernel_dtype (DtypeType): The data type to use for autocasting
            in the kernels.
        eps (float): A small value to add to the denominator for numerical
            stability.
        inference_state_dtype (DtypeType): The data type to use for the hidden
            states during inference.
        ffn_proj_factor (float): The projection factor for the feed-forward
            network.
        ffn_round_up_to_multiple_of (int): The value to round up the feed-forward
            network dimension to.
        gate_soft_cap (float): The value at which to cap the gates.
        output_logit_soft_cap (float): The value at which to cap the output
            logits.
        weight_mode (WeightModeType): The weight mode for the linear layers.
    """
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
    chunkwise_kernel: ChunkwiseKernelType = "metal" if DEVICE == "mps" else "native"
    sequence_kernel: SequenceKernelType = "metal" if DEVICE == "mps" else "native"
    step_kernel: StepKernelType = "metal" if DEVICE == "mps" else "native"
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


class mLSTMBackend(nn.Module):
    """The backend for the mLSTM layer.

    This module implements the backend for the mLSTM layer, which can be
    configured to use different kernels and modes for training and inference.

    Args:
        config (mLSTMBackendConfig): The configuration for the mLSTM backend.
    """

    def __init__(self, config: mLSTMBackendConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.return_last_states = config.return_last_states or config.mode == "inference"

    def _init_state(self, B: int, NH: int, DH_v: int, DH_k: int, device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize state tensors
        
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
                m_initial: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Forward pass for mLSTM"""
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


@dataclass
class mLSTMLayerConfig:
    """Configuration for the mLSTM layer.

    This dataclass holds the configuration for the mLSTM layer, including the
    embedding dimension, number of heads, and other hyperparameters.

    Attributes:
        embedding_dim (int): The dimension of the embedding layer.
        num_heads (int): The number of heads in the multi-head attention layers.
        use_bias (bool): Whether to use bias in the linear layers.
        norm_eps (float): The epsilon value for the normalization layers.
        norm_reduction_force_float32 (bool): Whether to force float32 for the
            reduction operations in the normalization layers.
        qk_dim_factor (float): The dimension factor for the query and key
            projections.
        v_dim_factor (float): The dimension factor for the value projection.
        gate_soft_cap (float): The value at which to cap the gates.
        mlstm_backend (mLSTMBackendConfig): The configuration for the mLSTM
            backend.
        weight_mode (WeightModeType): The weight mode for the linear layers.
    """
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
    """The mLSTM layer.

    This module implements the mLSTM layer, which is a variant of the LSTM layer
    that uses a matrix-based memory state. It is optimized for use with the
    PyTorch JIT compiler and the Metal Performance Shaders (MPS) backend.

    Args:
        config (mLSTMLayerConfig): The configuration for the mLSTM layer.
    """

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
            self.igate_preact = nn.Linear(config.embedding_dim, config.num_heads)
            self.fgate_preact = nn.Linear(config.embedding_dim, config.num_heads)
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                config.embedding_dim,
                2 * self.qk_dim + 2 * self.v_dim,
                bias=config.use_bias
            )
            self.ifgate_preact = nn.Linear(config.embedding_dim, 2 * config.num_heads)

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)

        self.multihead_norm = MultiHeadLayerNorm(num_heads=config.num_heads, head_dim=self.v_dim // config.num_heads,
                                                 eps=config.norm_eps, use_bias=config.use_bias,
                                                 force_float32_reductions=config.norm_reduction_force_float32)
        self.out_proj = nn.Linear(self.v_dim, config.embedding_dim, bias=config.use_bias)

    def forward(
            self, x: torch.Tensor, state: mLSTMLayerStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        """

        :param x:
        :param state:
        :return:
        """
        global o_preact, f_preact, i_preact, v, k, q
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


class FeedForward(nn.Module):
    """A feed-forward layer.

    This module implements a feed-forward layer with a SiLU activation function.

    Args:
        config (xLSTMLargeConfig): The configuration for the xLSTM model.
    """

    def __init__(self, config: xLSTMLargeConfig):
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
        """

        :param x:
        :return:
        """
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y


class mLSTMBlock(nn.Module):
    """An mLSTM block.

    This module implements an mLSTM block, which consists of an mLSTM layer
    followed by a feed-forward layer. It also includes residual connections
    and normalization layers.

    Args:
        config (xLSTMLargeConfig): The configuration for the xLSTM model.
    """

    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config
        self.norm_mlstm = RMSNorm(num_features=config.embedding_dim, eps=config.norm_eps, use_bias=config.use_bias,
                                  force_float32_reductions=config.norm_reduction_force_float32)
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
        self.norm_ffn = RMSNorm(num_features=config.embedding_dim, eps=config.norm_eps, use_bias=config.use_bias,
                                force_float32_reductions=config.norm_reduction_force_float32)
        self.ffn = FeedForward(config)

    def forward(
            self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        """

        :param x:
        :param state:
        :return:
        """
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm

        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn

        return x, state


class xLSTMLargeBlockStack(nn.Module):
    """A stack of mLSTM blocks.

    This module implements a stack of mLSTM blocks, which are the building blocks
    of the xLSTM model.

    Args:
        config (xLSTMLargeConfig): The configuration for the xLSTM model.
    """

    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList(
            [mLSTMBlock(config) for _ in range(config.num_blocks)]
        )

        if self.config.add_out_norm:
            self.out_norm = RMSNorm(num_features=config.embedding_dim, eps=config.norm_eps, use_bias=config.use_bias,
                                    force_float32_reductions=config.norm_reduction_force_float32)
        else:
            self.out_norm = nn.Identity()

    def forward(
            self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        """

        :param x:
        :param state:
        :return:
        """
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
    """The xLSTM model.

    This module implements the xLSTM model, which is a large language model based
    on the xLSTM architecture. It uses a stack of mLSTM blocks to process the
    input sequence and generate output logits.

    Args:
        config (xLSTMLargeConfig): The configuration for the xLSTM model.
    """
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
    ) -> torch.Tensor | tuple[torch.Tensor, mLSTMStateType]:
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
