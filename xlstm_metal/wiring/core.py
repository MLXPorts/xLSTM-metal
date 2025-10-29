"""
MAD Wiring Core - Backend-Agnostic Abstractions

Defines the core wiring system without any framework dependencies.
Backend-specific implementations provide their own WiredMADModel.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Supported compute backends"""
    MLX = "mlx"
    TORCH_COMPILED = "torch_compiled"
    TRITON = "triton"


class BlockType(Enum):
    """MAD block types"""
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
    Specification for a single MAD block.

    Attributes:
        name: Unique identifier for this block
        block_type: Type of block (MLSTM, FFN, etc.)
        backend: Compute backend (MLX, PyTorch, etc.)
        params: Dictionary of block-specific parameters
        polarity: Excitatory (+1.0) or inhibitory (-1.0) default polarity
    """
    name: str
    block_type: BlockType
    backend: BackendType = BackendType.MLX
    params: Dict = None
    polarity: float = 1.0

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class MADWiring:
    """
    Backend-agnostic MAD wiring graph.

    Manages block connectivity using adjacency matrix representation.
    Does NOT instantiate blocks - that's done by backend-specific WiredMADModel.
    """

    def __init__(self, block_specs: Dict[str, BlockSpec]):
        """
        Initialize wiring from block specifications.

        Args:
            block_specs: Dictionary mapping block names to BlockSpec
        """
        self.block_specs = block_specs
        self.block_names = list(block_specs.keys())
        self.num_blocks = len(self.block_names)

        # Map block names to indices
        self.name_to_idx = {name: i for i, name in enumerate(self.block_names)}

        # Adjacency matrix: adjacency[i][j] = polarity of connection from i to j
        # 0 = no connection, +1 = excitatory, -1 = inhibitory
        self.adjacency = [[0.0 for _ in range(self.num_blocks)] for _ in range(self.num_blocks)]

    def add_connection(
        self,
        source: str,
        target: str,
        polarity: Optional[float] = None
    ):
        """
        Add directed connection from source block to target block.

        Args:
            source: Source block name
            target: Target block name
            polarity: Connection polarity (+1.0 excitatory, -1.0 inhibitory)
                     If None, uses source block's default polarity
        """
        if source not in self.name_to_idx:
            raise ValueError(f"Source block '{source}' not found")
        if target not in self.name_to_idx:
            raise ValueError(f"Target block '{target}' not found")

        source_idx = self.name_to_idx[source]
        target_idx = self.name_to_idx[target]

        # Use source block's polarity if not specified
        if polarity is None:
            polarity = self.block_specs[source].polarity

        self.adjacency[source_idx][target_idx] = polarity

    def get_connections(
        self,
        block_name: str,
        direction: str = 'outgoing'
    ) -> List[str]:
        """
        Get connections for a block.

        Args:
            block_name: Name of block
            direction: 'outgoing' or 'incoming'

        Returns:
            List of connected block names
        """
        if block_name not in self.name_to_idx:
            raise ValueError(f"Block '{block_name}' not found")

        idx = self.name_to_idx[block_name]
        connections = []

        if direction == 'outgoing':
            # Find all j where adjacency[idx][j] != 0
            for j in range(self.num_blocks):
                if self.adjacency[idx][j] != 0:
                    connections.append(self.block_names[j])
        elif direction == 'incoming':
            # Find all i where adjacency[i][idx] != 0
            for i in range(self.num_blocks):
                if self.adjacency[i][idx] != 0:
                    connections.append(self.block_names[i])
        else:
            raise ValueError(f"Invalid direction '{direction}', must be 'incoming' or 'outgoing'")

        return connections

    def get_execution_stages(self) -> List[List[str]]:
        """
        Compute execution stages using topological sort (Kahn's algorithm).

        Blocks in the same stage can execute in parallel (no dependencies).

        Returns:
            List of stages, where each stage is a list of block names

        Raises:
            ValueError: If circular dependency detected
        """
        # Calculate in-degree for each block
        in_degree = [0] * self.num_blocks
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                if self.adjacency[i][j] != 0:
                    in_degree[j] += 1

        # Topological sort with stages
        stages = []
        visited = set()

        while len(visited) < self.num_blocks:
            # Find all blocks with in-degree 0 (no unprocessed dependencies)
            current_stage = []
            for i in range(self.num_blocks):
                if i not in visited and in_degree[i] == 0:
                    current_stage.append(self.block_names[i])

            if not current_stage:
                # Cycle detected or disconnected graph
                remaining = [self.block_names[i] for i in range(self.num_blocks) if i not in visited]
                raise ValueError(f"Circular dependency detected! Remaining blocks: {remaining}")

            stages.append(current_stage)

            # Mark current stage as visited
            for name in current_stage:
                idx = self.name_to_idx[name]
                visited.add(idx)

                # Decrease in-degree for all successors
                for successor_idx in range(self.num_blocks):
                    if self.adjacency[idx][successor_idx] != 0:
                        in_degree[successor_idx] -= 1

        return stages

    def visualize(self) -> str:
        """
        Generate text visualization of the wiring graph.

        Returns:
            ASCII art representation of connectivity
        """
        lines = ["MAD Wiring Graph", "=" * 60]

        for name in self.block_names:
            spec = self.block_specs[name]
            outgoing = self.get_connections(name, 'outgoing')

            lines.append(f"\n[{name}] ({spec.block_type.value}, {spec.backend.value})")

            if outgoing:
                for target in outgoing:
                    idx = self.name_to_idx[name]
                    target_idx = self.name_to_idx[target]
                    polarity = self.adjacency[idx][target_idx]
                    symbol = "+" if polarity > 0 else "-" if polarity < 0 else "·"
                    lines.append(f"  {symbol}---> {target}")
            else:
                lines.append("  (no outgoing connections)")

        lines.append("\n" + "=" * 60)
        lines.append("\nExecution Stages (parallel within stage):")
        for stage_idx, stage in enumerate(self.get_execution_stages()):
            lines.append(f"  Stage {stage_idx}: {', '.join(stage)}")

        return "\n".join(lines)
