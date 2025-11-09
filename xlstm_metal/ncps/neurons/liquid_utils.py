"""Utility functions for MLX liquid neural networks."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def lecun_tanh(x: mx.array) -> mx.array:
    """LeCun's scaled tanh activation."""
    return 1.7159 * mx.tanh(0.666 * x)


def sigmoid(x: mx.array) -> mx.array:
    """Sigmoid activation implemented with MLX ops."""
    return 1.0 / (1.0 + mx.exp(-x))


def get_activation(name: str) -> Callable[[mx.array], mx.array]:
    """Return an activation callable by name."""

    activations = {
        "lecun_tanh": lecun_tanh,
        "tanh": mx.tanh,
        "relu": nn.relu,
        "gelu": nn.gelu,
        "sigmoid": sigmoid,
        "identity": lambda x: x,
    }

    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(activations.keys())}"
        )

    return activations[name]
