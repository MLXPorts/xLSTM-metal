#!/usr/bin/env python
"""
NCPS Wiring Core - Neural Circuit Policies Base Classes

Following the NCPS (Neural Circuit Policies) architecture:
- Wiring: Defines connectivity patterns via adjacency matrices
- Clean separation between circuit design and layer implementation
- Backend-agnostic connectivity specifications
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math


class BackendType(Enum):
    """Supported compute backends"""
    MLX = "mlx"
    TORCH_COMPILED = "torch_compiled"
    TRITON = "triton"


class BlockType(Enum):
    """MAD block types (neuron types)"""
    MLSTM = "mlstm"
    SLSTM = "slstm"
    FFN = "ffn"
    LINEAR = "linear"
    NORM = "norm"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    TOKENIZER = "tokenizer"


@dataclass
class BlockSpec:
    """
    Specification for a single MAD block (neuron).

    Attributes:
        name: Unique identifier for this neuron
        block_type: Type of neuron (MLSTM, FFN, etc.)
        backend: Compute backend (MLX, PyTorch, etc.)
        params: Neuron-specific parameters
        polarity: Default connection polarity (+1 excitatory, -1 inhibitory)
    """
    name: str
    block_type: BlockType
    backend: BackendType = BackendType.MLX
    params: Dict[str, Any] = None
    polarity: float = 1.0

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class Wiring:
    """
    NCPS Wiring - Neural Circuit Connectivity Blueprint.

    Defines the connectivity pattern for neural circuits using adjacency matrices.
    Based on the NCPS (Neural Circuit Policies) architecture.

    Key concepts:
    - Adjacency matrix: Defines neuron-to-neuron connections
    - Sensory adjacency matrix: Defines input-to-neuron connections
    - Synapses have polarity: +1 (excitatory) or -1 (inhibitory)
    - Layers use wiring patterns to build their computation graphs
    """

    def __init__(self, units: int) -> None:
        """
        Initialize wiring blueprint.

        Args:
            units: Number of neurons in the circuit
        """
        self.units = units
        self.adjacency_matrix = self._create_matrix(units, units)
        self.sensory_adjacency_matrix: Optional[Any] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    def _create_matrix(self, rows: int, cols: int) -> Any:
        """Create adjacency matrix (backend-specific implementation)."""
        # Default to list of lists - backends can override
        return [[0 for _ in range(cols)] for _ in range(rows)]

    @property
    def num_layers(self) -> int:
        """Number of layers in this wiring topology."""
        return 1

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """Get neuron indices for a specific layer."""
        if layer_id == 0:
            return list(range(self.units))
        raise ValueError(f"Wiring has {self.num_layers} layers, got {layer_id}")

    def is_built(self) -> bool:
        """Check if wiring has been built with input dimensions."""
        return self.input_dim is not None

    def build(self, input_dim: int) -> None:
        """
        Build wiring with input connectivity.

        Args:
            input_dim: Input feature dimension
        """
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Conflicting input dimensions: expected {self.input_dim}, got {input_dim}"
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def set_input_dim(self, input_dim: int) -> None:
        """Set input dimension and initialize sensory connections."""
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = self._create_matrix(input_dim, self.units)

    def set_output_dim(self, output_dim: int) -> None:
        """Set output dimension."""
        self.output_dim = output_dim

    def get_type_of_neuron(self, neuron_id: int) -> str:
        """Get neuron type (inter/motor/command)."""
        if self.output_dim is None:
            return "inter"
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src: int, dest: int, polarity: int) -> None:
        """
        Add synapse between neurons.

        Args:
            src: Source neuron index
            dest: Destination neuron index
            polarity: +1 (excitatory) or -1 (inhibitory)
        """
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid source neuron {src} for {self.units} units")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        self._set_matrix_entry(self.adjacency_matrix, src, dest, polarity)

    def add_sensory_synapse(self, src: int, dest: int, polarity: int) -> None:
        """
        Add sensory synapse from input to neuron.

        Args:
            src: Input feature index
            dest: Destination neuron index
            polarity: +1 (excitatory) or -1 (inhibitory)
        """
        if self.input_dim is None or self.sensory_adjacency_matrix is None:
            raise ValueError("Cannot add sensory synapse before build().")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"Invalid sensory index {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        self._set_matrix_entry(self.sensory_adjacency_matrix, src, dest, polarity)

    def _set_matrix_entry(self, matrix: Any, row: int, col: int, value: int) -> None:
        """Set matrix entry (backend-specific implementation)."""
        matrix[row][col] = value

    def get_config(self) -> Dict[str, Any]:
        """Get wiring configuration for serialization."""
        return {
            "units": self.units,
            "adjacency_matrix": self._matrix_to_list(self.adjacency_matrix),
            "sensory_adjacency_matrix": None
            if self.sensory_adjacency_matrix is None
            else self._matrix_to_list(self.sensory_adjacency_matrix),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    def _matrix_to_list(self, matrix: Any) -> List[List[int]]:
        """Convert matrix to list format."""
        if hasattr(matrix, 'tolist'):
            return matrix.tolist()
        return matrix

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Wiring":
        """Create wiring from configuration."""
        wiring = cls(config["units"])
        wiring.adjacency_matrix = cls._list_to_matrix(config["adjacency_matrix"])
        if config.get("sensory_adjacency_matrix") is not None:
            wiring.sensory_adjacency_matrix = cls._list_to_matrix(config["sensory_adjacency_matrix"])
        wiring.input_dim = config.get("input_dim")
        wiring.output_dim = config.get("output_dim")
        return wiring

    @classmethod
    def _list_to_matrix(cls, data: List[List[int]]) -> Any:
        """Convert list to matrix format."""
        return data  # Default implementation

    def visualize(self) -> str:
        """
        Generate ASCII visualization of the neural circuit.

        Returns:
            String representation of connectivity
        """
        lines = ["NCPS Neural Circuit", "=" * 60]

        for name in range(self.units):
            neuron_type = self.get_type_of_neuron(name)
            lines.append(f"\n[{name}] ({neuron_type})")

            # Show outgoing connections
            outgoing = []
            for dest in range(self.units):
                if self.adjacency_matrix[name][dest] != 0:
                    polarity = self.adjacency_matrix[name][dest]
                    symbol = "+" if polarity > 0 else "-"
                    outgoing.append(f"{symbol}→ {dest}")

            if outgoing:
                lines.append("  " + ", ".join(outgoing))
            else:
                lines.append("  (no outgoing synapses)")

        lines.append("\n" + "=" * 60)
        lines.append(f"Layers: {self.num_layers}")
        for layer_idx in range(self.num_layers):
            neurons = self.get_neurons_of_layer(layer_idx)
            lines.append(f"  Layer {layer_idx}: {neurons}")

        return "\n".join(lines)


# Legacy MADWiring for backward compatibility
# TODO: Remove this once all code uses the new Wiring system
class MADWiring(Wiring):
    """
    Legacy MADWiring - deprecated, use Wiring instead.

    This class exists for backward compatibility during the transition
    to proper NCPS architecture.
    """

    def __init__(self, block_specs: Dict[str, BlockSpec]):
        """
        Initialize legacy MAD wiring.

        Args:
            block_specs: Dict mapping block names to BlockSpec objects
        """
        # Convert block specs to simple unit count
        super().__init__(units=len(block_specs))
        self.block_specs = block_specs
        self.block_names = list(block_specs.keys())
        self.name_to_idx = {name: idx for idx, name in enumerate(self.block_names)}

    def add_connection(self, from_block: str, to_block: str, polarity: float = 1.0):
        """Legacy method - use add_synapse instead."""
        from_idx = self.name_to_idx[from_block]
        to_idx = self.name_to_idx[to_block]
        self.add_synapse(from_idx, to_idx, int(polarity))

    def get_connections(self, block_name: str, direction: str = 'outgoing') -> List[str]:
        """Legacy method - use get_synapses instead."""
        idx = self.name_to_idx[block_name]
        indices = []
        matrix = self.adjacency_matrix

        if direction == 'outgoing':
            for j in range(self.units):
                if matrix[idx][j] != 0:
                    indices.append(j)
        elif direction == 'incoming':
            for i in range(self.units):
                if matrix[i][idx] != 0:
                    indices.append(i)

        return [self.block_names[i] for i in indices]

    def get_execution_stages(self) -> List[List[str]]:
        """Legacy method - compute topological stages."""
        # Simple implementation - could be more sophisticated
        return [self.block_names]

