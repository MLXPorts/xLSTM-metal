"""
sLSTM Block for xLSTM

Implements the scalar LSTM (sLSTM) layer from xLSTM paper.
Based on Appendix A equations 3-9: https://arxiv.org/pdf/2405.04517
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .components import MultiHeadLayerNorm, RMSNorm, soft_cap
from .kernel import slstm_sequential


@dataclass
class sLSTMConfig:
    """
    Configuration for sLSTM block matching xLSTM specification.

    sLSTM uses scalar recurrence with block-diagonal R matrices for efficiency.
    Each head processes independently with its own recurrent connections.

    Key parameters:
        embedding_dim: Model dimension (e.g. 4096)
        num_heads: Number of heads (e.g. 8)
        head_dim: Dimension per head (embedding_dim // num_heads)
        gate_soft_cap: Soft cap for gate pre-activations (default: 15.0)
    """
    embedding_dim: int = 4096
    num_heads: int = 8
    gate_soft_cap: float = 15.0
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    eps: float = 1e-6
    inference_state_dtype: str = "float32"
    return_last_states: bool = True

    def __post_init__(self):
        assert self.embedding_dim % self.num_heads == 0, \
            f"embedding_dim={self.embedding_dim} must be divisible by num_heads={self.num_heads}"
        self.head_dim = self.embedding_dim // self.num_heads


class sLSTMLayer(nn.Module):
    """
    sLSTM Layer with block-diagonal recurrent connections.

    Weight structure:
        Linear projections (input to gates):
            igate.weight: [num_heads, embedding_dim]
            fgate.weight: [num_heads, embedding_dim]
            zgate.weight: [embedding_dim, embedding_dim]
            ogate.weight: [embedding_dim, embedding_dim]

        Recurrent projections (block-diagonal):
            recurrent_kernel: [num_heads, head_dim, 4 * head_dim]
                This is block-diagonal: one block per head
                Each block connects head state to all 4 gates for that head

        Normalization:
            group_norm.weight: [embedding_dim]

        Biases (optional):
            igate.bias, fgate.bias, zgate.bias, ogate.bias
    """

    def __init__(self, config: sLSTMConfig):
        super().__init__()
        self.config = config

        # Input projections for gates
        # Note: Input and forget gates project to num_heads (per-head gates)
        self.igate = nn.Linear(config.embedding_dim, config.num_heads)
        self.fgate = nn.Linear(config.embedding_dim, config.num_heads)

        # Cell input and output gates project to full embedding_dim
        self.zgate = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.use_bias
        )
        self.ogate = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.use_bias
        )

        # Block-diagonal recurrent kernel
        # Shape: [num_heads, head_dim, 4 * head_dim]
        # Each head has its own recurrent block connecting to i,f,z,o gates
        self.recurrent_kernel = mx.random.normal(
            shape=(config.num_heads, config.head_dim, 4 * config.head_dim)
        ) * 0.01  # Small initialization

        # Group normalization (applied per-head)
        self.group_norm = MultiHeadLayerNorm(
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32
        )

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass through sLSTM layer.

        Args:
            x: Input tensor [B, S, D]
            state: Optional tuple (c_state, n_state, m_state)
                c_state: Cell state [B, NH, H]
                n_state: Normalizer state [B, NH, H]
                m_state: Stabilizer [B, NH]

        Returns:
            output: Output tensor [B, S, D]
            new_state: Updated (c_state, n_state, m_state) if return_last_states
        """
        B, S, D = x.shape
        NH = self.config.num_heads
        H = self.config.head_dim

        # Compute input projections
        # i,f gates: [B, S, NH]
        i_preact = self.igate(x)
        f_preact = self.fgate(x)

        # z,o gates: [B, S, D]
        z_preact = self.zgate(x)
        o_preact = self.ogate(x)

        # Apply soft cap to gates
        i_preact = soft_cap(i_preact, self.config.gate_soft_cap)
        f_preact = soft_cap(f_preact, self.config.gate_soft_cap)
        o_preact = soft_cap(o_preact, self.config.gate_soft_cap)

        # Reshape z_preact and o_preact to [B, S, NH, H]
        z_shaped = z_preact.reshape(B, S, NH, H)
        o_shaped = o_preact.reshape(B, S, NH, H)

        # Apply activation to cell input: z = tanh(z̃)  (eq 11)
        z = mx.tanh(z_shaped)  # [B, S, NH, H]

        # Add recurrent connections if we have previous state
        if state is not None:
            c_prev, n_prev, m_prev = state
            # Get previous hidden state: h_{t-1} from c and n
            # h_{t-1} ≈ c_{t-1} / (n_{t-1} + eps)
            h_prev = c_prev / (n_prev + self.config.eps)  # [B, NH, H]

            # Apply block-diagonal recurrent kernel
            # h_prev: [B, NH, H] -> reshape to [B, NH, 1, H]
            # recurrent_kernel: [NH, H, 4*H]
            # Result: [B, NH, 4*H]
            h_prev_expanded = h_prev[:, :, None, :]  # [B, NH, 1, H]
            recurrent = mx.matmul(h_prev_expanded, self.recurrent_kernel)  # [B, NH, 1, 4*H]
            recurrent = recurrent.squeeze(2)  # [B, NH, 4*H]

            # Split recurrent into gate contributions
            # Each gate gets head_dim features
            r_i, r_f, r_z, r_o = mx.split(recurrent, 4, axis=-1)  # Each [B, NH, H]

            # Add recurrent contributions to gates
            # i,f are already per-head [B, S, NH], r_i/r_f are [B, NH, H] -> reduce over H
            i_preact += r_i.mean(axis=-1)[:, None, :]  # [B, S, NH]
            f_preact += r_f.mean(axis=-1)[:, None, :]  # [B, S, NH]

            # z,o need to be reshaped: r_z/r_o are [B, NH, H]
            z += r_z[:, None, :, :]  # [B, S, NH, H]
            o_shaped += r_o[:, None, :, :]  # [B, S, NH, H]

        # Reshape for kernel: [B, NH, S, H]
        z = z.transpose(0, 2, 1, 3)  # [B, NH, S, H]
        o_shaped = o_shaped.transpose(0, 2, 1, 3)  # [B, NH, S, H]

        # Reshape gates: [B, NH, S]
        i_preact = i_preact.transpose(0, 2, 1)  # [B, NH, S]
        f_preact = f_preact.transpose(0, 2, 1)  # [B, NH, S]

        # Reshape o_preact for per-head processing
        o_preact_reshaped = o_shaped  # Already [B, NH, S, H]

        # Extract initial states
        c_initial, n_initial, m_initial = (None, None, None)
        if state is not None:
            c_initial, n_initial, m_initial = state

        # Run sLSTM kernel
        h_out, new_state = slstm_sequential(
            z=z,
            i_preact=i_preact,
            f_preact=f_preact,
            o_preact=o_preact_reshaped.mean(axis=-1),  # Reduce to [B, NH, S]
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            eps=self.config.eps,
            return_last_states=self.config.return_last_states
        )  # h_out: [B, NH, S, H]

        # Transpose for group normalization: [B, NH, S, H] -> [B, S, NH, H]
        h_out = h_out.transpose(0, 2, 1, 3)  # [B, S, NH, H]

        # Apply group normalization
        h_out = self.group_norm(h_out)  # [B, S, D] (flattened output)

        # Reshape back to [B, S, D] (already done by group_norm)
        output = h_out

        return output, new_state


class sLSTMBlock(nn.Module):
    """
    Complete sLSTM block with normalization.

    Structure:
        x -> RMSNorm -> sLSTMLayer -> residual -> output
    """

    def __init__(self, config: sLSTMConfig):
        super().__init__()
        self.config = config

        # Pre-normalization
        self.norm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32
        )

        # sLSTM layer
        self.slstm_layer = sLSTMLayer(config)

    def __call__(
            self,
            x: mx.array,
            state: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        """
        Forward pass with pre-norm and residual connection.

        Args:
            x: Input tensor [B, S, D]
            state: Optional sLSTM state

        Returns:
            output: Output tensor [B, S, D]
            new_state: Updated state
        """
        # Pre-normalization
        x_norm = self.norm(x)

        # sLSTM layer
        h, new_state = self.slstm_layer(x_norm, state)

        # Residual connection
        output = x + h

        return output, new_state
