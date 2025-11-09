"""MLX implementation of the Liquid Time-Constant recurrent module."""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .. import wirings

from .ltc_cell import LTCCell


class LTC(nn.Module):
    def __init__(
        self,
        input_size: int,
        units: Union[int, wirings.Wiring],
        return_sequences: bool = True,
        batch_first: bool = True,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = True,
    ) -> None:
        super().__init__()

        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, wirings.Wiring):
            wiring = units
        else:
            wiring = wirings.FullyConnected(units)
        wiring.build(input_size)

        self._wiring = wiring
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
        )
        self.state_size = wiring.units
        self.output_size = wiring.output_dim

    # ------------------------------------------------------------------ #
    def synapse_count(self) -> int:
        return int(mx.sum(mx.abs(mx.array(self._wiring.adjacency_matrix))))

    def sensory_synapse_count(self) -> int:
        return int(mx.sum(mx.abs(mx.array(self._wiring.sensory_adjacency_matrix))))

    def apply_weight_constraints(self) -> None:
        self.rnn_cell.apply_weight_constraints()

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        inputs: mx.array,
        hx: Optional[mx.array] = None,
        timespans: Optional[mx.array] = None,
    ):
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

            h_out, h_state = self.rnn_cell(step_input, h_state, ts)
            if self.return_sequences:
                outputs.append(h_out)

        if self.return_sequences:
            readout = mx.stack(outputs, axis=seq_dim)
        else:
            readout = h_out

        if not is_batched:
            readout = mx.squeeze(readout, axis=batch_dim)
            h_state = mx.squeeze(h_state, axis=0)

        return readout, h_state
