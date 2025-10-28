#!/usr/bin/env python
"""
xLSTM Layer with NCPS Wiring Support

Following NCPS architecture:
- Layer takes Wiring object as parameter
- Wiring defines connectivity pattern via adjacency matrices
- Layer instantiates cells per neuron and applies connectivity
- Parameters are MLX arrays automatically tracked by nn.Module
"""

from typing import Optional, Tuple, Union, List
import mlx.core as mx
import mlx.nn as nn

from ..core import Wiring
from ...blocks.mlstm.xlstm_block import xLSTMBlock, xLSTMBlockConfig


class xLSTMLayer(nn.Module):
    """
    xLSTM Layer with neural circuit wiring.

    Similar to WiredCfCCell in NCPS, this layer:
    1. Takes a Wiring object defining connectivity
    2. Creates xLSTM cells for each neuron in the wiring
    3. Applies connectivity pattern during forward pass
    4. Manages hierarchical states across layers

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(32, 4)  # 32 neurons, 4 outputs
        >>> layer = xLSTMLayer(input_size=20, wiring=wiring, config=xlstm_config)
        >>> output, state = layer(input, state)
    """

    def __init__(
        self,
        input_size: Optional[int],
        wiring: Union[int, Wiring],
        config: Optional[xLSTMBlockConfig] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
    ):
        """
        Initialize xLSTM layer with wiring.

        Args:
            input_size: Input feature dimension
            wiring: Either an integer (number of units) or a Wiring object
            config: xLSTM configuration (required if wiring is not provided)
            return_sequences: Whether to return all timesteps or just last
            batch_first: Whether batch dimension is first
        """
        super().__init__()

        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # Handle wiring setup
        if isinstance(wiring, int):
            # Simple case: just specify number of units
            # Create default sequential wiring
            from .xlstm_7b import xLSTMWiring
            wiring = xLSTMWiring(num_blocks=wiring)

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Provide 'input_size' or call wiring.build()."
            )

        self._wiring = wiring
        self._cells: List[xLSTMBlock] = []

        # Create xLSTM cells for each layer in the wiring
        in_features = wiring.input_dim
        for layer_idx in range(wiring.num_layers):
            hidden_units = wiring.get_neurons_of_layer(layer_idx)

            # Get sparsity mask from wiring
            if layer_idx == 0:
                # First layer: sensory input connectivity
                input_sparsity = wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                # Later layers: inter-neuron connectivity
                prev_neurons = wiring.get_neurons_of_layer(layer_idx - 1)
                input_sparsity = wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_neurons, :]

            # Add recurrent connections (each neuron connects to itself)
            input_sparsity = mx.concatenate(
                [
                    mx.array(input_sparsity, dtype=mx.int32),
                    mx.ones((len(hidden_units), len(hidden_units)), dtype=mx.int32),
                ],
                axis=0,
            )

            # Create xLSTM cell for this layer with configured parameters
            if config is None:
                raise ValueError("xLSTMBlockConfig must be provided")

            # Create a config for this specific layer
            layer_config = xLSTMBlockConfig(
                embedding_dim=len(hidden_units),
                num_heads=config.num_heads,
                qk_dim_factor=config.qk_dim_factor,
                v_dim_factor=config.v_dim_factor,
                gate_soft_cap=config.gate_soft_cap,
                ffn_proj_factor=config.ffn_proj_factor,
                ffn_act_fn=config.ffn_act_fn,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                norm_reduction_force_float32=config.norm_reduction_force_float32,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
                return_last_states=config.return_last_states,
                chunk_size=config.chunk_size
            )

            cell = xLSTMBlock(layer_config)
            self._cells.append(cell)

            # Register cell as module attribute for proper parameter tracking
            setattr(self, f"layer_{layer_idx}", cell)

            in_features = len(hidden_units)

    @property
    def state_size(self) -> int:
        """Total number of neurons in the circuit."""
        return self._wiring.units

    @property
    def layer_sizes(self) -> List[int]:
        """Number of neurons in each layer."""
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self) -> int:
        """Number of layers in the wiring."""
        return self._wiring.num_layers

    @property
    def output_size(self) -> int:
        """Output dimension."""
        return self._wiring.output_dim if self._wiring.output_dim else self._wiring.units

    @property
    def synapse_count(self) -> mx.array:
        """Total number of synapses in the circuit."""
        adj = mx.array(self._wiring.adjacency_matrix, dtype=mx.int32)
        return mx.sum(mx.abs(adj))

    @property
    def sensory_synapse_count(self) -> mx.array:
        """Number of sensory synapses (input to neurons)."""
        if self._wiring.sensory_adjacency_matrix is None:
            return mx.array(0, dtype=mx.int32)
        sens_adj = mx.array(self._wiring.sensory_adjacency_matrix, dtype=mx.int32)
        return mx.sum(mx.abs(sens_adj))

    def __call__(
        self,
        inputs: mx.array,
        hx: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass through xLSTM neural circuit.

        Args:
            inputs: Input tensor [B, S, input_dim] or [S, B, input_dim]
            hx: Hidden state [B, state_size] (optional)

        Returns:
            output: Output tensor [B, S, output_dim] or [S, B, output_dim]
            new_state: Updated hidden state [B, state_size]
        """
        # Handle batch_first
        if not self.batch_first:
            inputs = mx.transpose(inputs, (1, 0, 2))  # [S, B, D] -> [B, S, D]

        batch_size, seq_len, _ = inputs.shape

        # Initialize hidden state if not provided
        if hx is None:
            hx = mx.zeros((batch_size, self.state_size))

        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]  # [B, input_dim]

            # Process through layers following wiring connectivity
            new_states = []
            offset = 0

            for size, cell in zip(self.layer_sizes, self._cells):
                # Extract current layer's state
                current_state = hx[:, offset:offset + size]

                # Forward through cell
                # xLSTMBlock expects [B, S, D] so add time dimension
                x_t_expanded = mx.expand_dims(x_t, axis=1)  # [B, 1, D]

                h, state = cell(x_t_expanded, current_state)

                # Remove time dimension
                h = mx.squeeze(h, axis=1)  # [B, D]

                # Update input for next layer
                x_t = h
                new_states.append(h)

                offset += size

            # Concatenate states across layers
            hx = mx.concatenate(new_states, axis=1)

            outputs.append(h)  # Output from last layer

        # Stack outputs
        output = mx.stack(outputs, axis=1)  # [B, S, output_dim]

        # Return only last output if not returning sequences
        if not self.return_sequences:
            output = output[:, -1, :]  # [B, output_dim]

        # Handle batch_first
        if not self.batch_first:
            output = mx.transpose(output, (1, 0, 2))  # [B, S, D] -> [S, B, D]

        return output, hx

