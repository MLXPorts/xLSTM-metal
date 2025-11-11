"""MLX implementation of the GatedFFN recurrent-style module.

This follows the NCPS pattern where a cell (GatedFFNCell) is wrapped
by a module that handles batching and sequence processing.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .gated_ffn_cell import GatedFFNCell


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network module following NCPS patterns.
    
    Wraps GatedFFNCell to handle batch and sequence dimensions,
    similar to how CfC wraps CfCCell.
    
    Args:
        input_size: Input dimension (embedding_dim)
        hidden_size: Intermediate dimension (proj_up_dim)
        proj_size: Optional output projection size
        return_sequences: If True, return all timesteps; if False, return only last
        batch_first: If True, input shape is [B, S, D]; if False, [S, B, D]
        activation: Activation function name ('silu', 'gelu', 'relu')
        use_bias: Whether to use bias in linear layers
        dropout: Optional dropout rate
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
        """
        Forward pass through the GatedFFN module.
        
        Args:
            inputs: Input tensor [B, S, D] if batch_first else [S, B, D]
            hx: Hidden state (unused for FFN, kept for API compatibility)
            
        Returns:
            (output, state): Output tensor and state (None for stateless FFN)
        """
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
