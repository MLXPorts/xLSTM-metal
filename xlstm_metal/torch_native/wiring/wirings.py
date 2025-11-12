"""PyTorch wiring utilities for constructing sparse neural circuits."""

from typing import Dict, List, Optional

import torch


def _set_matrix_entry(matrix: torch.Tensor, row: int, col: int, value: int) -> None:
    matrix[row, col] = value


class Wiring:
    """Connectivity blueprint describing synapses between neurons.

    Uses torch tensors internally so downstream code can stay in the same framework.
    """

    def __init__(self, units: int) -> None:
        self.units = units
        self.adjacency_matrix = torch.zeros((units, units), dtype=torch.int8)
        self.sensory_adjacency_matrix: Optional[torch.Tensor] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    @property
    def num_layers(self) -> int:
        return 1

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        return list(range(self.units))

    def is_built(self) -> bool:
        return self.input_dim is not None

    def build(self, input_dim: int) -> None:
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided: expected "
                f"{self.input_dim}, got {input_dim}."
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, *_: object, **__: object) -> torch.Tensor:
        return self.adjacency_matrix.clone()

    def sensory_erev_initializer(self, *_: object, **__: object) -> torch.Tensor:
        if self.sensory_adjacency_matrix is None:
            raise ValueError("Sensory adjacency matrix not initialised.")
        return self.sensory_adjacency_matrix.clone()

    def set_input_dim(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = torch.zeros((input_dim, self.units), dtype=torch.int8)

    def set_output_dim(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def get_type_of_neuron(self, neuron_id: int) -> str:
        if self.output_dim is None:
            return "inter"
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src: int, dest: int, polarity: int = 1) -> None:
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid source neuron {src} for {self.units} units")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        _set_matrix_entry(self.adjacency_matrix, src, dest, polarity)

    def add_sensory_synapse(self, src: int, dest: int, polarity: int = 1) -> None:
        if self.input_dim is None or self.sensory_adjacency_matrix is None:
            raise ValueError("Cannot add sensory synapse before build().")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"Invalid sensory index {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        _set_matrix_entry(self.sensory_adjacency_matrix, src, dest, polarity)

    def get_config(self) -> Dict[str, object]:
        return {
            "units": self.units,
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "sensory_adjacency_matrix": None
            if self.sensory_adjacency_matrix is None
            else self.sensory_adjacency_matrix.tolist(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "Wiring":
        wiring = cls(config['units'])
        wiring.adjacency_matrix = torch.tensor(config["adjacency_matrix"], dtype=torch.int8)
        if config.get("sensory_adjacency_matrix") is not None:
            wiring.sensory_adjacency_matrix = torch.tensor(
                config["sensory_adjacency_matrix"], dtype=torch.int8
            )
        wiring.input_dim = config.get("input_dim")
        wiring.output_dim = config.get("output_dim")
        return wiring

    @property
    def synapse_count(self) -> int:
        return int(torch.sum(torch.abs(self.adjacency_matrix)).item())

    @property
    def sensory_synapse_count(self) -> int:
        if self.sensory_adjacency_matrix is None:
            return 0
        return int(torch.sum(torch.abs(self.sensory_adjacency_matrix)).item())

__all__ = ["Wiring"]
