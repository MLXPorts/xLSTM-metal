#!/usr/bin/env python
"""
xLSTM Wiring for NCPS-Style Neural Circuits

Following the NCPS (Neural Circuit Policies) architecture where:
- Wiring defines connectivity patterns (adjacency matrices)
- Layers use wiring patterns to build their computation graphs
- Clean separation between circuit design and execution
"""

from typing import Dict, Any, Optional, List
from .core import Wiring, BackendType


class xLSTMWiring(Wiring):
    """
    xLSTM Neural Circuit Wiring.

    Defines the connectivity pattern for xLSTM blocks in a neural circuit.
    Follows NCPS architecture with sequential block connections.
    """

    def __init__(self, num_blocks: int):
        """
        Initialize xLSTM wiring.

        Args:
            num_blocks: Number of xLSTM blocks in sequence
        """
        # Each xLSTM block is treated as a single "neuron" in the circuit
        super().__init__(units=num_blocks)
        self.num_blocks = num_blocks

    def build(self, input_dim: int) -> None:
        """
        Build the wiring with input connectivity.

        Args:
            input_dim: Input feature dimension
        """
        super().build(input_dim)

        # xLSTM has sequential connectivity: block_0 -> block_1 -> ... -> block_N
        # All blocks receive input from the previous block (or external input for block_0)
        # This is a simple sequential pattern, but could be extended to more complex topologies

    @property
    def num_layers(self) -> int:
        """xLSTM has all blocks in a single sequential layer."""
        return 1

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """All xLSTM blocks are in layer 0."""
        if layer_id == 0:
            return list(range(self.units))
        raise ValueError(f"xLSTM wiring has only 1 layer, got {layer_id}")

    def get_type_of_neuron(self, neuron_id: int) -> str:
        """All neurons are xLSTM blocks."""
        return "xlstm"


class AutoNCPxLSTMWiring(Wiring):
    """
    AutoNCP-inspired xLSTM wiring with hierarchical structure.

    Uses AutoNCP principles to create a more sophisticated xLSTM circuit:
    - Inter blocks: Standard xLSTM processing
    - Command blocks: Control/attention mechanisms
    - Motor blocks: Output/final processing
    """

    def __init__(
        self,
        total_blocks: int,
        output_blocks: int = 1,
        sparsity_level: float = 0.7,
        seed: int = 22222
    ):
        """
        Initialize AutoNCP xLSTM wiring.

        Args:
            total_blocks: Total number of xLSTM blocks
            output_blocks: Number of output/final blocks
            sparsity_level: Connection sparsity (0.0 = fully connected, 1.0 = minimal connections)
            seed: Random seed for reproducibility
        """
        if output_blocks >= total_blocks:
            raise ValueError(f"output_blocks ({output_blocks}) must be < total_blocks ({total_blocks})")

        super().__init__(units=total_blocks)
        self.set_output_dim(output_blocks)

        # AutoNCP-inspired distribution
        inter_and_command = total_blocks - output_blocks
        command_blocks = max(int(0.4 * inter_and_command), 1)
        inter_blocks = inter_and_command - command_blocks

        self._inter_blocks = list(range(inter_blocks))
        self._command_blocks = list(range(inter_blocks, inter_blocks + command_blocks))
        self._motor_blocks = list(range(inter_blocks + command_blocks, total_blocks))

        self._sparsity_level = sparsity_level
        self._seed = seed

        # Build connectivity pattern
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """Build the AutoNCP connectivity pattern."""
        import random
        rng = random.Random(self._seed)

        # Inter -> Command connections (with sparsity)
        density = 1.0 - self._sparsity_level
        for inter_idx in self._inter_blocks:
            for command_idx in self._command_blocks:
                if rng.random() < density:
                    polarity = 1 if rng.random() > 0.5 else -1
                    self.add_synapse(inter_idx, command_idx, polarity)

        # Command -> Motor connections (with sparsity)
        for command_idx in self._command_blocks:
            for motor_idx in self._motor_blocks:
                if rng.random() < density:
                    polarity = 1 if rng.random() > 0.5 else -1
                    self.add_synapse(command_idx, motor_idx, polarity)

        # Recurrent connections within command layer (inspired by NCP)
        recurrent_density = density * 0.5  # Sparser recurrent connections
        for src in self._command_blocks:
            for dest in self._command_blocks:
                if src != dest and rng.random() < recurrent_density:
                    polarity = 1 if rng.random() > 0.5 else -1
                    self.add_synapse(src, dest, polarity)

    @property
    def num_layers(self) -> int:
        """AutoNCP xLSTM has 3 layers: inter, command, motor."""
        return 3

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """Return neurons for each layer."""
        if layer_id == 0:
            return self._inter_blocks
        elif layer_id == 1:
            return self._command_blocks
        elif layer_id == 2:
            return self._motor_blocks
        raise ValueError(f"AutoNCP xLSTM has 3 layers, got {layer_id}")

    def get_type_of_neuron(self, neuron_id: int) -> str:
        """Return neuron type based on layer."""
        if neuron_id in self._inter_blocks:
            return "inter"
        elif neuron_id in self._command_blocks:
            return "command"
        elif neuron_id in self._motor_blocks:
            return "motor"
        raise ValueError(f"Unknown neuron {neuron_id}")


def create_xlstm_wiring(config: Dict[str, Any]) -> Wiring:
    """
    Create NCPS-style xLSTM wiring from configuration.

    Args:
        config: Configuration dict with architectural parameters

    Returns:
        Wiring object defining the xLSTM neural circuit
    """
    num_blocks = config['num_blocks']

    # For now, use simple sequential wiring
    # Could be extended to use AutoNCPxLSTMWiring for more sophisticated patterns
    wiring = xLSTMWiring(num_blocks)

    # Build with input dimension (will be set when used in a layer)
    # wiring.build(input_dim) will be called by the layer

    return wiring
