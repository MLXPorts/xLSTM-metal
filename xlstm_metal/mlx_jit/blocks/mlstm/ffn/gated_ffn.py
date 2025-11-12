"""Gated Feed-Forward Network Module – MLX Implementation (NCPS Sequence Wrapper)

Overview
--------
GatedFFN is a sequence-processing wrapper around GatedFFNCell, following
the NCPS pattern where:
  - **Cell**: Single-step computation with parameters (GatedFFNCell)
  - **Module**: Batch/sequence handling wrapper (this class)

This separation enables flexible integration with NCPS wiring while maintaining
compatibility with standard sequence-to-sequence interfaces.

NCPS Module Pattern
-------------------
Similar to how RNN modules wrap RNN cells:
  - Cell: processes one timestep at a time
  - Module: iterates over sequence dimension, calling cell repeatedly

For stateless FFN, this pattern seems redundant but maintains API consistency
with stateful cells (LSTM, GRU, CfC) in NCPS frameworks.

Sequence Processing Modes
--------------------------
1. **return_sequences=True** (default):
   - Returns all timestep outputs [B, S, D]
   - Used for encoder-style processing

2. **return_sequences=False**:
   - Returns only last timestep [B, D]
   - Used for sequence classification

Batch Dimension Ordering
-------------------------
- **batch_first=True** (default): Input shape [B, S, D]
  - Standard PyTorch/MLX convention
  - B = batch size, S = sequence length, D = features

- **batch_first=False**: Input shape [S, B, D]
  - Legacy RNN convention (rarely used in modern code)

When to Use This vs GatedFFNCell Directly?
-------------------------------------------
Use **GatedFFNCell** when:
  - You have already-batched single-step inputs [B, D]
  - You're building custom sequence processing logic
  - You want minimal overhead

Use **GatedFFN** when:
  - You need standard sequence-to-sequence interface
  - You want compatibility with NCPS wiring infrastructure
  - You need return_sequences or batch_first options

In Practice
-----------
For xLSTM blocks, FFN is typically applied to entire sequences at once
(not iteratively), so direct GatedFFNCell usage is more efficient:

  # Efficient (used in xLSTM blocks)
  ffn_cell = GatedFFNCell(input_size, hidden_size)
  output, _ = ffn_cell(x)  # x: [B, S, D]

  # Equivalent but slower (iterates over sequence)
  ffn_module = GatedFFN(input_size, hidden_size)
  output, _ = ffn_module(x)  # x: [B, S, D]

This wrapper is primarily for NCPS-style applications where the wiring
framework expects a module-level interface.

Optional Projection
-------------------
If `proj_size` is specified, an additional linear layer projects the output
to a different dimension. This is useful for encoder-decoder architectures
or when embedding dimension differs from model dimension.

Stateless Property
------------------
Unlike LSTM/GRU, FFN has no hidden state. The `hx` parameter and return
value are kept for API compatibility but are always None.

Parity
------
Logic mirrors torch-native GatedFFN for cross-backend testing.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .gated_ffn_cell import GatedFFNCell


class GatedFFN(nn.Module):
    """Sequence-processing wrapper for GatedFFNCell (NCPS module pattern).

    Handles batch and sequence dimensions, iterates over timesteps calling
    the underlying cell. Primarily for NCPS framework compatibility.

    Parameters
    ----------
    input_size : int
        Input dimension (embedding_dim).
    hidden_size : int
        Intermediate dimension for FFN (~2.667 * input_size typical).
    proj_size : int | None, optional
        Optional output projection dimension (default: None = input_size).
    return_sequences : bool, default True
        Whether to return all timesteps (True) or only last (False).
    batch_first : bool, default True
        Whether input shape is [B, S, D] (True) or [S, B, D] (False).
    activation : {"silu", "gelu", "relu"}, default "silu"
        Activation function for gating.
    use_bias : bool, default False
        Whether linear layers include bias.
    dropout : float | None, optional
        Dropout probability for regularization.

    Returns (forward)
    -----------------
    output : mx.array
        - If return_sequences=True: [B, S, output_size] (or [S, B, output_size])
        - If return_sequences=False: [B, output_size] (or [output_size])
    state : None
        Always None (FFN is stateless).

    Examples
    --------
    >>> ffn = GatedFFN(input_size=512, hidden_size=1365)
    >>> x = mx.random.normal((4, 32, 512))  # [B, S, D]
    >>> y, state = ffn(x)
    >>> y.shape
    (4, 32, 512)
    >>> state is None
    True
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: Optional[int] = None,
            return_sequences: bool = True,
            batch_first: bool = True,
            activation: str = "silu",
            use_bias: bool = False,
            dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.state_size = None  # FFN is stateless
        self.output_size = input_size

        # Create the cell
        self.rnn_cell = GatedFFNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            use_bias=use_bias,
            dropout=dropout,
        )

        # Optional projection layer
        if proj_size is None:
            self._fc = None
        else:
            self._fc = nn.Linear(self.output_size, proj_size)
            self.output_size = proj_size

    def _apply_fc(self, tensor: mx.array) -> mx.array:
        """Apply optional projection layer."""
        if self._fc is None:
            return tensor
        return self._fc(tensor)

    def __call__(
            self,
            inputs: mx.array,
            hx: Optional[mx.array] = None
    ) -> tuple[mx.array, Optional[mx.array]]:
        """Process sequence through gated FFN.

        Parameters
        ----------
        inputs : mx.array
            Input sequences [B, S, D] if batch_first else [S, B, D].
        hx : mx.array | None, optional
            Hidden state (unused, kept for API compatibility).

        Returns
        -------
        output : mx.array
            Processed sequences (shape depends on return_sequences).
        state : None
            Always None (stateless).

        Notes
        -----
        Iterates over sequence dimension, applying cell at each timestep.
        For efficiency, consider using GatedFFNCell directly on full
        sequences when return_sequences=True and no special processing needed.
        """
        global h_out
        is_batched = inputs.ndim == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0

        # Add batch dimension if needed
        if not is_batched:
            inputs = mx.expand_dims(inputs, axis=batch_dim)

        batch_size = inputs.shape[batch_dim]
        seq_len = inputs.shape[seq_dim]

        # FFN is stateless, so no hidden state initialization needed
        h_state = None

        outputs = []
        for t in range(seq_len):
            if self.batch_first:
                step_input = inputs[:, t, :]
            else:
                step_input = inputs[t, :, :]

            # Process through the cell
            h_out, h_state = self.rnn_cell(step_input, h_state)

            if self.return_sequences:
                outputs.append(self._apply_fc(h_out))

        # Prepare output
        if self.return_sequences:
            stacked = mx.stack(outputs, axis=seq_dim)
            readout = stacked
        else:
            readout = self._apply_fc(h_out)

        # Remove batch dimension if input wasn't batched
        if not is_batched:
            readout = mx.squeeze(readout, axis=batch_dim)

        return readout, h_state


__all__ = ['GatedFFN']
