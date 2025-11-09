"""MLX implementation of the CfC recurrent module."""

from typing import Callable, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .. import wirings

from .cfc_cell import CfCCell
from .wired_cfc_cell import WiredCfCCell


class CfC(nn.Module):
    def __init__(
        self,
        input_size: int,
        units: Union[int, wirings.Wiring],
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[float] = None,
        custom_activations: Optional[Dict[str, Callable[[], nn.Module]]] = None,
    ) -> None:
        super().__init__()

        if mixed_memory:
            raise NotImplementedError("mixed_memory is not supported in the MLX port yet.")

        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, wirings.Wiring):
            wiring = units
            wiring.build(input_size)
            self._wiring = wiring
            self.state_size = wiring.units
            self.output_size = wiring.output_dim
            self.rnn_cell = WiredCfCCell(
                input_size,
                wiring,
                mode=mode,
                activation=activation,
                custom_activations=custom_activations,
            )
        else:
            hidden_units = units
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            self.state_size = hidden_units
            self.output_size = hidden_units
            self.rnn_cell = CfCCell(
                input_size,
                hidden_units,
                mode=mode,
                activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                custom_activations=custom_activations,
            )

        if proj_size is None:
            self._fc = None
        else:
            self._fc = nn.Linear(self.output_size, proj_size)
            self.output_size = proj_size

    # ------------------------------------------------------------------ #
    def _apply_fc(self, tensor: mx.array) -> mx.array:
        if self._fc is None:
            return tensor
        return self._fc(tensor)

    # ------------------------------------------------------------------ #
    def __call__(self, inputs: mx.array, hx: Optional[mx.array] = None, timespans=None):
        is_batched = inputs.ndim == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0

        if not is_batched:
            inputs = mx.expand_dims(inputs, axis=batch_dim)
            if timespans is not None:
                timespans = mx.expand_dims(timespans, axis=batch_dim)

        batch_size = inputs.shape[batch_dim]
        seq_len = inputs.shape[seq_dim]

        if hx is None:
            h_state = mx.zeros((batch_size, self.state_size), dtype=mx.float32)
        else:
            if not is_batched and hx.ndim == 1:
                h_state = mx.expand_dims(hx, axis=0)
            else:
                h_state = hx

        outputs = []
        for t in range(seq_len):
            if self.batch_first:
                step_input = inputs[:, t, :]
                ts = 1.0 if timespans is None else timespans[:, t]
            else:
                step_input = inputs[t, :, :]
                ts = 1.0 if timespans is None else timespans[t, :]

            if hasattr(self.rnn_cell, "layer_sizes"):
                h_out, h_state = self.rnn_cell(step_input, h_state, ts)
            else:
                h_out, h_state = self.rnn_cell(step_input, h_state, ts)

            if self.return_sequences:
                outputs.append(self._apply_fc(h_out))

        if self.return_sequences:
            stacked = mx.stack(outputs, axis=seq_dim)
            readout = stacked
        else:
            readout = self._apply_fc(h_out)

        if not is_batched:
            readout = mx.squeeze(readout, axis=batch_dim)
            h_state = mx.squeeze(h_state, axis=0)

        return readout, h_state
