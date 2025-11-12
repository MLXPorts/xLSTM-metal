"""Gated Feed-Forward Network Cell – MLX Implementation (SwiGLU Pattern)

Overview
--------
GatedFFNCell implements the SwiGLU (Swish-Gated Linear Unit) feed-forward
network, a modern variant of FFN that uses gating to control information flow.
This is the standard FFN architecture in LLaMA, PaLM, and xLSTM models.

SwiGLU Architecture
-------------------
Traditional FFN:
  FFN(x) = W₂ · ReLU(W₁ · x)

SwiGLU (Gated FFN):
  FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)

where ⊙ denotes element-wise multiplication and SiLU(x) = x · σ(x).

The gating mechanism allows the network to learn which features to amplify
or suppress, improving expressiveness without additional depth.

Why Gated FFN?
--------------
1. **Better Expressiveness**: Gating provides multiplicative interactions
2. **Smoother Gradients**: SiLU has smoother derivatives than ReLU
3. **Empirical Performance**: SwiGLU outperforms ReLU FFN in LLMs
4. **Parameter Efficiency**: Same param count as 2-layer FFN with larger hidden

NCPS Pattern
------------
This cell follows the NCPS (Neural Circuit Policies) pattern:
  - Cell = single-step computation with all trainable parameters
  - Module wraps cell for sequence/batch processing
  - Stateless: returns (output, None) since FFN has no hidden state

Typical Usage in xLSTM
-----------------------
In xLSTM blocks, GatedFFN is applied after mLSTM/sLSTM:
  x = layer_norm(x)
  x = mLSTM(x)
  x = x + residual

  residual = x
  x = layer_norm(x)
  x = GatedFFN(x)
  x = x + residual

Activation Functions
--------------------
Supported activations:
  - **silu** (default): SiLU(x) = x · σ(x), smooth and non-monotonic
  - **gelu**: GELU(x) ≈ x · Φ(x), used in BERT/GPT
  - **relu**: ReLU(x) = max(0, x), traditional choice

SiLU is the standard for SwiGLU and provides best empirical results.

Sparsity Mask
-------------
Optional `sparsity_mask` enables structured sparsity patterns for NCPS
wiring. When provided, the mask is applied element-wise to the gated hidden
states, zeroing out specific connections.

Dropout
-------
Optional dropout is applied to the final output for regularization during
training. Typically disabled for inference.

Parameters vs FLOPs
-------------------
For embedding_dim D and hidden_size H:
  - Parameters: 3 * D * H (gate, up, down projections)
  - FLOPs: ~6 * D * H * S (forward pass for sequence length S)

Typical hidden_size is 2.667 * embedding_dim (xLSTM-7B uses this ratio).

Parity
------
Logic mirrors torch-native GatedFFNCell for cross-backend testing.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class GatedFFNCell(nn.Module):
    """SwiGLU-style gated FFN cell (stateless, single-step).

    Implements gated feed-forward transformation with configurable activation.
    Follows NCPS cell pattern: encapsulates all parameters, single-step forward.

    Parameters
    ----------
    input_size : int
        Input/output dimension (embedding_dim).
    hidden_size : int
        Intermediate dimension (typically ~2.667 * input_size).
    activation : {"silu", "gelu", "relu"}, default "silu"
        Activation function for gating.
    use_bias : bool, default False
        Whether linear layers include bias (typically False for xLSTM).
    dropout : float | None, optional
        Dropout probability for output (training regularization).
    sparsity_mask : mx.array | None, optional
        Optional mask for structured sparsity (NCPS wiring).

    Returns (forward)
    -----------------
    output : mx.array [B, S, input_size]
        Transformed features.
    state : None
        Always None (FFN is stateless).

    Examples
    --------
    >>> cell = GatedFFNCell(input_size=4096, hidden_size=10880)
    >>> x = mx.random.normal((2, 64, 4096))
    >>> y, state = cell(x)
    >>> y.shape
    (2, 64, 4096)
    >>> state is None
    True
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            activation: str = "silu",
            use_bias: bool = False,
            dropout: Optional[float] = None,
            sparsity_mask: Optional[mx.array] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.activation = activation

        # Separate gate and value projections
        self.proj_up_gate = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.proj_up = nn.Linear(input_size, hidden_size, bias=use_bias)

        # Down projection
        self.proj_down = nn.Linear(hidden_size, input_size, bias=use_bias)

        # Activation
        if activation == "silu":
            self.act_fn = nn.silu
        elif activation == "gelu":
            self.act_fn = nn.gelu
        elif activation == "relu":
            self.act_fn = nn.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

        # Sparsity mask (for NCPS wiring)
        if sparsity_mask is not None:
            mask = mx.array(sparsity_mask) if not isinstance(sparsity_mask, mx.array) else sparsity_mask
            self._sparsity_mask = mx.abs(mask)
        else:
            self._sparsity_mask = None

    def __call__(self, x: mx.array, state: Optional[mx.array] = None) -> tuple[mx.array, Optional[mx.array]]:
        """Apply SwiGLU gated FFN transformation.

        Parameters
        ----------
        x : mx.array [B, S, input_size]
            Input features.
        state : mx.array | None, optional
            Unused (kept for NCPS API compatibility with stateful cells).

        Returns
        -------
        output : mx.array [B, S, input_size]
            Gated FFN output.
        state : None
            Always None (stateless).

        Notes
        -----
        Computation flow:
          1. gate = act(W_gate @ x)
          2. up = W_up @ x
          3. hidden = gate ⊙ up
          4. output = W_down @ hidden
        """
        # Separate gate and value projections
        gate = self.proj_up_gate(x)  # [B, S, hidden_size]
        z = self.proj_up(x)  # [B, S, hidden_size]

        # Gating: act(gate) * z
        gated = mx.multiply(self.act_fn(gate), z)

        # Apply sparsity mask if present
        if self._sparsity_mask is not None:
            gated = mx.multiply(gated, self._sparsity_mask)

        # Project down
        y = self.proj_down(gated)  # [B, S, input_size]

        # Apply dropout
        if self.dropout is not None:
            y = self.dropout(y)

        # Return output and state (None for stateless FFN)
        return y, None

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout": self.dropout.p if self.dropout else None,
            "sparsity_mask": self._sparsity_mask,
        }


__all__ = ['GatedFFNCell']
