
"""
xLSTM implementation for MLX (Apple Silicon GPU)

Overview
- Single-process, GPU-accelerated xLSTM using MLX arrays and ops.
- Optional tiled Metal GEMM for the final projection head (enable via
  `XLSTM_MLX_FAST_HEAD=1`) from `mlx_fast_kernels.gemm_kernels`.
- Designed to run prefill and decode on a dedicated MLX GPU stream.

References
- Beck et al. (2024): xLSTM: Extended Long Short-Term Memory.
- MLX Streams: see `tools/mlx_streams.py` and `docs/MLX_IMPLEMENTATION_GUIDE.md`.
"""

import os
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List



class CausalConv1d(nn.Module):
    """A causal 1D convolution layer for MLX.

    This layer implements a 1D convolution that preserves causality. It does this
    by adding padding to the input before the convolution and then trimming the
    output to remove the padded future region. This is useful for sequence
    modeling tasks where the output at each timestep should only depend on the
    inputs from previous timesteps.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        dilation (int, optional): The dilation rate of the kernel. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)
    
    def __call__(self, x):
        # MLX expects shape: (batch, length, channels)
        # x comes in as (batch, channels) or (batch, length, channels)
        if len(x.shape) == 2:
            # Add length dimension
            x = mx.expand_dims(x, axis=1)
        
        out = self.conv(x)
        # Remove future positions for causality
        if self.padding > 0:
            out = out[:, :-self.padding, :]
        return out


def block_diag(*matrices):
    """Create a block diagonal matrix from a list of matrices.

    This function takes a list of matrices and creates a block diagonal matrix
    from them.

    Args:
        *matrices: A list of matrices to form the block diagonal matrix.

    Returns:
        A block diagonal matrix.
    """
    rows = sum(mat.shape[0] for mat in matrices)
    cols = sum(mat.shape[1] for mat in matrices)
    
    result = mx.zeros((rows, cols))
    
    current_row = 0
    current_col = 0
    for mat in matrices:
        r, c = mat.shape
        result[current_row:current_row + r, current_col:current_col + c] = mat
        current_row += r
        current_col += c
    
    return result


class HeadLinear(nn.Module):
    """Per-head linear without constructing a block-diagonal matrix.

    Keeps weights as (H, Do, Di) and applies y[b,h,do] = sum_i x[b,h,i]*W[h,do,i].
    Accepts flattened input (B, H*Di) and returns flattened output (B, H*Do).
    """
    def __init__(self, num_heads: int, in_per_head: int, out_per_head: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.in_per_head = in_per_head
        self.out_per_head = out_per_head
        scale = mx.array(0.02) / mx.sqrt(in_per_head)
        # Weight: (H, Do, Di)
        self.weight = mx.random.normal((num_heads, out_per_head, in_per_head)) * scale
        self.bias = mx.zeros((num_heads, out_per_head)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H*Di) or (B, H, Di)
        if len(x.shape) == 2:
            B, F = int(x.shape[0]), int(x.shape[1])
            H, Di = self.num_heads, self.in_per_head
            assert F == H * Di, "HeadLinear: input last dim must equal H*Di"
            xh = x.reshape(B, H, Di)
        else:
            xh = x
            B, H, Di = int(xh.shape[0]), int(xh.shape[1]), int(xh.shape[2])
        # Broadcast multiply and sum over Di
        # xh: (B,H,Di) -> (B,H,1,Di); W: (H,Do,Di) -> (1,H,Do,Di)
        y = mx.sum(xh[:, :, None, :] * self.weight[None, :, :, :], axis=-1)  # (B,H,Do)
        if self.bias is not None:
            y = y + self.bias[None, :, :]
        return y.reshape(B, H * self.out_per_head)


class MultiHeadLayerNorm(nn.Module):
    """Head-aware LayerNorm: normalize per head over DH, then flatten.

    Uses float32 reductions for stability; optional affine scale/bias.
    """
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = mx.ones((num_heads * head_dim)) if affine else None
        self.bias = mx.zeros((num_heads * head_dim)) if affine else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, NH*DH)
        B, F = int(x.shape[0]), int(x.shape[1])
        NH, DH = self.num_heads, self.head_dim
        assert F == NH * DH, "MultiHeadLayerNorm: feature dim must be NH*DH"
        xh = x.reshape(B, NH, DH)
        in_dtype = xh.dtype
        xf = xh.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.var(xf, axis=-1, keepdims=True)
        xnorm = (xf - mean) * mx.rsqrt(var + self.eps)
        y = xnorm.astype(in_dtype).reshape(B, F)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


def soft_cap(x: mx.array, cap: float) -> mx.array:
    capv = mx.array(cap, dtype=x.dtype)
    return capv * mx.tanh(x / capv)


def enlarge_as(x, target):
    """Expand tensor x to match the number of dimensions of a target tensor.

    This function adds new dimensions to the end of the input tensor `x` until
    it has the same number of dimensions as the `target` tensor.

    Args:
        x (mx.array): The input tensor.
        target (mx.array): The target tensor.

    Returns:
        mx.array: The expanded tensor.
    """
    while len(x.shape) < len(target.shape):
        x = mx.expand_dims(x, axis=-1)
    return x


class sLSTMBlock(nn.Module):
    """A scalar LSTM (sLSTM) block with exponential gating and state normalization.

    This block implements the sLSTM variant from the xLSTM paper. It uses
    grouped normalization and an optional causal convolution on the input.

    Args:
        inp_dim (int): The input dimension.
        head_dim (int): The dimension of each head.
        head_num (int): The number of heads.
        p_factor (float, optional): The projection factor for the up-projection.
            Defaults to 4/3.
        ker_size (int, optional): The kernel size for the causal convolution.
            Defaults to 4.
    """
    def __init__(self, inp_dim, head_dim, head_num, p_factor=4/3, ker_size=4):
        super().__init__()
        self.inp_dim = inp_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim = head_dim * head_num
        
        self.inp_norm = nn.LayerNorm(inp_dim)
        self.hid_norm = MultiHeadLayerNorm(head_num, head_dim)
        
        # Conv1d for sLSTM - operates on inp_dim channels
        self.causal_conv = CausalConv1d(inp_dim, inp_dim, kernel_size=ker_size)
        
        self.W_z = nn.Linear(inp_dim, self.hidden_dim)
        self.W_i = nn.Linear(inp_dim, self.hidden_dim)
        self.W_o = nn.Linear(inp_dim, self.hidden_dim)
        self.W_f = nn.Linear(inp_dim, self.hidden_dim)
        
        self.R_z = HeadLinear(head_num, head_dim, head_dim)
        self.R_i = HeadLinear(head_num, head_dim, head_dim)
        self.R_o = HeadLinear(head_num, head_dim, head_dim)
        self.R_f = HeadLinear(head_num, head_dim, head_dim)
        
        proj_dim = int(p_factor * self.hidden_dim)
        self.up_proj = nn.Linear(self.hidden_dim, 2 * proj_dim)
        self.down_proj = nn.Linear(proj_dim, inp_dim)
    
    def init_hidden(self, batch_size):
        """Initialize hidden states"""
        n_0 = mx.ones((batch_size, self.hidden_dim))
        c_0 = mx.zeros((batch_size, self.hidden_dim))
        h_0 = mx.zeros((batch_size, self.hidden_dim))
        m_0 = mx.zeros((batch_size, self.hidden_dim))
        return c_0, n_0, h_0, m_0
    
    def __call__(self, x, hidden_state, use_conv=False):
        c_tm1, n_tm1, h_tm1, m_tm1 = hidden_state
        
        x_t = self.inp_norm(x)
        
        if use_conv:
            # Apply causal convolution
            x_c = self.causal_conv(x_t)
            x_c = nn.silu(mx.squeeze(x_c, axis=1) if len(x_c.shape) > 2 else x_c)
        else:
            x_c = x_t
        
        i_t = soft_cap(self.W_i(x_c) + self.R_i(h_tm1), 15.0)
        f_t = soft_cap(self.W_f(x_c) + self.R_f(h_tm1), 15.0)
        z_t = self.W_z(x_t) + self.R_z(h_tm1)
        o_t = self.W_o(x_t) + self.R_o(h_tm1)
        
        m_t = mx.maximum(f_t + m_tm1, i_t)
        i_t = mx.exp(i_t - m_t)
        f_t = mx.exp(f_t - m_t + m_tm1)
        
        z_t = mx.tanh(z_t)
        o_t = mx.sigmoid(o_t)
        
        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t
        h_t = o_t * (c_t / mx.maximum(n_t, mx.array(1.0)))
        
        out = self.hid_norm(h_t)
        out1, out2 = mx.split(self.up_proj(out), 2, axis=-1)
        out = out1 * nn.gelu(out2)
        out = self.down_proj(out)
        
        return out + x, (c_t, n_t, h_t, m_t)


class mLSTMBlock(nn.Module):
    """A matrix LSTM (mLSTM) block with a covariance update rule.

    This block implements the mLSTM variant from the xLSTM paper. It maintains a
    per-head covariance-like state that is updated with exponential gates. It
    projects inputs via left and right paths and mixes them with a causal
    convolution for local smoothing before computing queries, keys, values, and
    gating.

    Args:
        inp_dim (int): The input dimension.
        head_dim (int): The dimension of each head.
        head_num (int): The number of heads.
        p_factor (float, optional): The projection factor for the up-projection.
            Defaults to 2.
        ker_size (int, optional): The kernel size for the causal convolution.
            Defaults to 4.
    """
    def __init__(self, inp_dim, head_dim, head_num, p_factor=2, ker_size=4):
        super().__init__()
        self.inp_dim = inp_dim
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim = head_dim * head_num
        
        self.inp_norm = nn.LayerNorm(inp_dim)
        self.hid_norm = MultiHeadLayerNorm(head_num, head_dim)
        
        self.up_l_proj = nn.Linear(inp_dim, int(p_factor * inp_dim))
        self.up_r_proj = nn.Linear(inp_dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, inp_dim)
        
        # Conv1d for mLSTM - operates on projected dimension
        proj_dim_int = int(p_factor * inp_dim)
        self.causal_conv = CausalConv1d(proj_dim_int, proj_dim_int, kernel_size=ker_size)
        self.skip_connection = nn.Linear(proj_dim_int, self.hidden_dim)
        
        self.W_i = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_f = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_o = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        
        self.W_q = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        self.W_k = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
        self.W_v = nn.Linear(int(p_factor * inp_dim), self.hidden_dim)
    
    def init_hidden(self, batch_size):
        """Initialize hidden states"""
        c_0 = mx.zeros((batch_size, self.head_num, self.head_dim, self.head_dim))
        n_0 = mx.ones((batch_size, self.head_num, self.head_dim))
        m_0 = mx.zeros((batch_size, self.head_num))
        return c_0, n_0, m_0
    
    def __call__(self, x, hidden_state):
        bs = x.shape[0]
        c_tm1, n_tm1, m_tm1 = hidden_state
        
        x_n = self.inp_norm(x)
        
        x_t = self.up_l_proj(x_n)
        r_t = self.up_r_proj(x_n)
        
        # Apply causal convolution
        x_c = self.causal_conv(x_t)
        x_c = nn.silu(mx.squeeze(x_c, axis=1) if len(x_c.shape) > 2 else x_c)
        x_skip = self.skip_connection(x_c)
        
        q_t = mx.reshape(self.W_q(x_c), (bs, self.head_num, self.head_dim))
        k_t = mx.reshape(self.W_k(x_c), (bs, self.head_num, self.head_dim)) / mx.sqrt(self.head_dim)
        v_t = mx.reshape(self.W_v(x_t), (bs, self.head_num, self.head_dim))
        
        i_t = soft_cap(self.W_i(x_c), 15.0)
        f_t = soft_cap(self.W_f(x_c), 15.0)
        o_t = mx.sigmoid(self.W_o(x_t))
        
        m_t = mx.maximum(f_t + m_tm1, i_t)
        i_t = mx.exp(i_t - m_t)
        f_t = mx.exp(f_t - m_t + m_tm1)
        
        # Covariance update
        i_expanded = mx.expand_dims(mx.expand_dims(i_t, axis=-1), axis=-1)
        f_expanded = mx.expand_dims(mx.expand_dims(f_t, axis=-1), axis=-1)
        v_expanded = mx.expand_dims(v_t, axis=-1)
        k_expanded = mx.expand_dims(k_t, axis=-2)
        
        c_t = f_expanded * c_tm1 + i_expanded * (v_expanded @ k_expanded)
        
        f_n_expanded = mx.expand_dims(f_t, axis=-1)
        i_n_expanded = mx.expand_dims(i_t, axis=-1)
        n_t = f_n_expanded * n_tm1 + i_n_expanded * k_t
        
        # Compute output
        q_expanded = mx.expand_dims(q_t, axis=-1)
        h_numerator = mx.squeeze(c_t @ q_expanded, axis=-1)  # (batch, head_num, head_dim)
        h_denominator = mx.maximum(mx.sum(n_t * q_t, axis=-1, keepdims=True), mx.array(1.0))  # (batch, head_num, 1)
        h_t_heads = h_numerator / h_denominator  # (batch, head_num, head_dim)
        h_t_flat = mx.reshape(h_t_heads, (bs, self.hidden_dim))  # Flatten heads
        h_t = o_t * h_t_flat  # Apply output gate
        
        out = self.hid_norm(h_t) + x_skip
        out = out * nn.silu(r_t)
        out = self.down_proj(out)
        
        return out + x, (c_t, n_t, m_t)


class xLSTM(nn.Module):
    """An xLSTM model that combines sLSTM and mLSTM blocks.

    This model consists of an embedding layer, followed by a series of sLSTM and
    mLSTM blocks, and finally an output projection layer. The sLSTM and mLSTM
    blocks are alternated according to the `signature` argument.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_layers (int): The total number of sLSTM and mLSTM blocks.
        signature (Tuple[int, int]): A tuple specifying the number of mLSTM and
            sLSTM blocks in the repeating pattern.
        inp_dim (int): The input and embedding dimension.
        head_dim (int): The dimension of each head.
        head_num (int): The number of heads.
        p_factor (Tuple[float, float], optional): A tuple containing the
            projection factors for the mLSTM and sLSTM blocks. Defaults to (2.0, 4/3).
        ker_size (int, optional): The kernel size for the causal convolution.
            Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
    """
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        signature: Tuple[int, int],  # (num_mLSTM, num_sLSTM)
        inp_dim: int,
        head_dim: int,
        head_num: int,
        p_factor: Tuple[float, float] = (2.0, 4/3),  # (mLSTM_factor, sLSTM_factor)
        ker_size: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.inp_dim = inp_dim
        self.hidden_dim = head_dim * head_num
        
        self.embedding = nn.Embedding(vocab_size, inp_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        m_factor, s_factor = p_factor
        m_num, s_num = signature
        block_types = [True] * m_num + [False] * s_num
        
        # Cycle through block types for the specified number of layers
        self.blocks = []
        for i in range(num_layers):
            block_type = block_types[i % len(block_types)]
            if block_type:  # mLSTM block
                self.blocks.append(mLSTMBlock(inp_dim, head_dim, head_num, m_factor, ker_size))
            else:  # sLSTM block
                self.blocks.append(sLSTMBlock(inp_dim, head_dim, head_num, s_factor, ker_size))
        
        self.head = nn.Linear(inp_dim, vocab_size)
        # Fast head toggle (runtime-configurable) — default off for safety/training
        self.use_fast_head = False
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for all blocks"""
        return [block.init_hidden(batch_size) for block in self.blocks]
    
    def __call__(self, tokens, hidden_states=None, return_hidden=False):
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
        
        # Process sequence through blocks
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, block in enumerate(self.blocks):
                x_t, hidden_states[i] = block(x_t, hidden_states[i])
                if self.dropout and i < len(self.blocks) - 1:
                    x_t = self.dropout(x_t)
            outputs.append(x_t)
        
        # Stack outputs and compute logits
        output = mx.stack(outputs, axis=1)
        # Compute logits; optional fast head matmul using MLX Metal kernels
        # Order of precedence: self.use_fast_head (if explicitly set) > runtime config > env fallback
        use_fast_head = None
        from tools.mlx_runtime import get_runtime_config as _get_runtime_config  # type: ignore
        rc = _get_runtime_config()
        if rc.get("fast_head") is not None:
            use_fast_head = bool(rc.get("fast_head"))
        if use_fast_head is None:
            use_fast_head = self.use_fast_head if self.use_fast_head is not None else (os.environ.get("XLSTM_MLX_FAST_HEAD", "1") == "1")
        if use_fast_head:
            try:
                from mlx_src.mlx_fast_kernels.gemm_kernels import gemm_av
                # Flatten (B, T, D) -> (B*T, D) and use W^T for (D, V)
                bt, d = int(output.shape[0] * output.shape[1]), int(output.shape[2])
                v = int(self.vocab_size)
                A = output.reshape(bt, d).astype(mx.float32)
                Wt = mx.transpose(self.head.weight.astype(mx.float32))  # (D, V)
                logits2d = gemm_av(A, Wt)  # (B*T, V)
                if getattr(self.head, "bias", None) is not None:
                    logits2d = logits2d + self.head.bias.astype(logits2d.dtype)
                logits = logits2d.reshape(output.shape[0], output.shape[1], v)
            except Exception:
                # Fallback to standard linear on any error
                logits = self.head(output)
        else:
            logits = self.head(output)
        # Output logit soft-cap for numerical stability
        logits = soft_cap(logits, 30.0)

        if return_hidden:
            return logits, hidden_states
        return logits

    def set_fast_head(self, enabled: bool) -> None:
        """Enable/disable fast head projection (tiled GEMM) for inference.

        This does not change weights; it only switches the final projection
        implementation. For training with autograd, prefer `enabled=False`.
        """
        self.use_fast_head = bool(enabled)


def create_xlstm_model(
    vocab_size: int = 50257,
    num_layers: int = 12,
    signature: Tuple[int, int] = (7, 1),
    inp_dim: int = 768,
    head_dim: int = 96,
    head_num: int = 8,
    p_factor: Tuple[float, float] = (2.0, 4/3),
    ker_size: int = 4,
    dropout: float = 0.1
) -> xLSTM:
    """
    Create an xLSTM model with specified configuration
    
    Args:
        vocab_size: Size of vocabulary
        num_layers: Number of xLSTM blocks
        signature: (num_mLSTM, num_sLSTM) blocks in pattern
        inp_dim: Input/embedding dimension
        head_dim: Dimension per head
        head_num: Number of heads
        p_factor: Projection factors for (mLSTM, sLSTM)
        ker_size: Kernel size for causal convolution
        dropout: Dropout rate
        
    Returns:
        xLSTM model instance
    """
    return xLSTM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        signature=signature,
        inp_dim=inp_dim,
        head_dim=head_dim,
        head_num=head_num,
        p_factor=p_factor,
        ker_size=ker_size,
        dropout=dropout
    )


if __name__ == "__main__":
    # Example usage
    model = create_xlstm_model(
        vocab_size=1000,
        num_layers=4,
        signature=(1, 1),
        inp_dim=256,
        head_dim=32,
        head_num=8
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    tokens = mx.random.randint(0, 1000, (batch_size, seq_len))
    
    logits = model(tokens)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, 1000)")
