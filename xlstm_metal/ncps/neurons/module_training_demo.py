"""Minimal MLX training loop demo for unit tests and documentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class TrainingState:
    model: nn.Module
    optimizer: optim.Optimizer


def create_training_state(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    learning_rate: float = 1e-2,
) -> TrainingState:
    """Build a tiny two-layer network and its optimiser."""

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.relu,
        nn.Linear(hidden_dim, output_dim),
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    return TrainingState(model=model, optimizer=optimizer)


def _loss_fn(mdl: nn.Module, inputs: mx.array, targets: mx.array) -> mx.array:
    predictions = mdl(inputs)
    return mx.mean((predictions - targets) ** 2)


def train_for_epochs(
    state: TrainingState,
    dataset: Iterable[Tuple[mx.array, mx.array]],
    epochs: int,
) -> Iterator[float]:
    """Yield epoch losses for a simple supervised loop."""

    value_and_grad = nn.value_and_grad(state.model, _loss_fn)

    for _ in range(epochs):
        total_loss = mx.array(0.0, dtype=mx.float32)
        count = 0
        for inputs, targets in dataset:
            loss, grads = value_and_grad(state.model, inputs, targets)
            state.optimizer.update(state.model, grads)
            mx.eval(state.model.parameters(), state.optimizer.state)
            total_loss = total_loss + loss
            count += 1
        denom = mx.array(max(1, count), dtype=mx.float32)
        mean_loss = total_loss / denom
        yield mean_loss
