"""Recurrent wrapper for the LTC4Cell."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .ltc4_cell import LTC4Cell


class LTC4(nn.Module):
    def __init__(
        self,
        input_size: int,
        units: int,
        solver: str = "semi_implicit",
        ode_unfolds: int = 6,
        activation: str = "tanh",
        return_sequences: bool = True,
        return_state: bool = False,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.cell = LTC4Cell(
            units=units,
            solver=solver,
            ode_unfolds=ode_unfolds,
            activation=activation,
            input_dim=input_size,
        )
        self._input_size = input_size

    def __call__(
        self,
        inputs: mx.array,
        hx: Optional[mx.array] = None,
        timespans: Optional[mx.array] = None,
    ):
        del timespans  # original training ignores time deltas
        is_batched = inputs.ndim == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0

        if not is_batched:
            inputs = mx.expand_dims(inputs, axis=batch_dim)

        batch_size = inputs.shape[batch_dim]
        seq_len = inputs.shape[seq_dim]

        if hx is None:
            h_state = mx.zeros((batch_size, self.cell.units), dtype=mx.float32)
        else:
            h_state = hx if is_batched or hx.ndim == 2 else mx.expand_dims(hx, axis=0)

        outputs = []
        for t in range(seq_len):
            step_input = inputs[:, t, :] if self.batch_first else inputs[t, :, :]
            h_state, _ = self.cell(step_input, h_state)
            if self.return_sequences:
                outputs.append(h_state)

        if self.return_sequences:
            readout = mx.stack(outputs, axis=seq_dim)
        else:
            readout = h_state

        if not is_batched:
            readout = mx.squeeze(readout, axis=batch_dim)
            h_state = mx.squeeze(h_state, axis=0)

        if self.return_state:
            return readout, h_state
        return readout
