"""Wired CfC cell for MLX, mirroring the original Torch implementation."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .cfc_cell import CfCCell


class WiredCfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        mode: str = "default",
        activation: str = "lecun_tanh",
        custom_activations: Optional[Dict[str, Callable[[], nn.Module]]] = None,
    ) -> None:
        super().__init__()

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Provide 'input_size' or call wiring.build()."
            )

        self._wiring = wiring
        self._layers: List[CfCCell] = []

        in_features = wiring.input_dim
        for layer_idx in range(wiring.num_layers):
            hidden_units = wiring.get_neurons_of_layer(layer_idx)

            if layer_idx == 0:
                input_sparsity = wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_neurons = wiring.get_neurons_of_layer(layer_idx - 1)
                input_sparsity = wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_neurons, :]

            input_sparsity = mx.concatenate(
                [
                    input_sparsity,
                    mx.ones((len(hidden_units), len(hidden_units)), dtype=mx.int32),
                ],
                axis=0,
            )

            cell = CfCCell(
                in_features,
                len(hidden_units),
                mode=mode,
                activation=activation,
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
                custom_activations=custom_activations,
            )
            self._layers.append(cell)
            setattr(self, f"layer_{layer_idx}", cell)
            in_features = len(hidden_units)

    @property
    def state_size(self) -> int:
        return self._wiring.units

    @property
    def layer_sizes(self) -> List[int]:
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self) -> int:
        return self._wiring.num_layers

    @property
    def sensory_size(self) -> int:
        return self._wiring.input_dim

    @property
    def motor_size(self) -> int:
        return self._wiring.output_dim

    @property
    def output_size(self) -> int:
        return self.motor_size

    @property
    def synapse_count(self) -> int:
        # Device scalar (no host conversion)
        return mx.sum(mx.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self) -> int:
        # Device scalar (no host conversion)
        return mx.sum(mx.abs(self._wiring.adjacency_matrix))

    def __call__(self, inputs: mx.array, hx: mx.array, timespans) -> tuple[mx.array, mx.array]:
        new_states = []
        x = inputs
        offset = 0
        for size, layer in zip(self.layer_sizes, self._layers):
            current = hx[:, offset : offset + size]
            h, _ = layer(x, current, timespans)
            x = h
            new_states.append(h)
            offset += size

        new_h_state = mx.concatenate(new_states, axis=1)
        return h, new_h_state
