#!/usr/bin/env python
"""
MAD Wiring System - NCPS-Style Graph Composition

Replaces backends.py with declarative connectivity patterns.
Based on Neural Circuit Policies (NCP) wiring from Lechner et al.

Key Concepts:
- Adjacency matrix-based connectivity (sparse synaptic patterns)
- Polarity: +1 (excitatory), -1 (inhibitory), 0 (no connection)
- Topological execution ordering (parallel where dependencies allow)
- Backend-agnostic block composition
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn


class BackendType(Enum):
    """Available compute backends"""
    MLX = "mlx"
    TORCH = "torch"
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


@dataclass
class BlockSpec:
    """
    Specification for a single MAD block.

    Attributes:
        name: Unique identifier for this block
        block_type: Type of neural block (mlstm, slstm, ffn, etc.)
        backend: Compute backend (mlx, torch, torch_compiled)
        params: Block-specific parameters (d_model, num_heads, etc.)
        polarity: Connection polarity (+1 excitatory, -1 inhibitory)
    """
    name: str
    block_type: BlockType
    backend: BackendType = BackendType.MLX
    params: Dict[str, Any] = None
    polarity: float = 1.0  # +1 excitatory, -1 inhibitory

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class MADWiring:
    """
    Neural Circuit Wiring for MAD blocks.

    Based on NCPS (Lechner et al.):
    - Sparse adjacency matrix defines connectivity
    - Polarity (excitatory/inhibitory) encoded in weights
    - Topological ordering for parallel execution

    Example:
        >>> # Define blocks
        >>> specs = {
        ...     'input_norm': BlockSpec('input_norm', BlockType.NORM, params={'d_model': 512}),
        ...     'mlstm_1': BlockSpec('mlstm_1', BlockType.MLSTM, backend=BackendType.MLX),
        ...     'mlstm_2': BlockSpec('mlstm_2', BlockType.MLSTM, backend=BackendType.MLX),
        ...     'combiner': BlockSpec('combiner', BlockType.LINEAR, params={'in_features': 1024})
        ... }
        >>>
        >>> # Create wiring
        >>> wiring = MADWiring(specs)
        >>>
        >>> # Add connections (parallel heads)
        >>> wiring.add_connection('input_norm', 'mlstm_1')
        >>> wiring.add_connection('input_norm', 'mlstm_2')
        >>> wiring.add_connection('mlstm_1', 'combiner')
        >>> wiring.add_connection('mlstm_2', 'combiner')
        >>>
        >>> # Get execution stages (mlstm_1 and mlstm_2 run in parallel)
        >>> stages = wiring.get_execution_stages()
        >>> # [[input_norm], [mlstm_1, mlstm_2], [combiner]]
    """

    def __init__(self, block_specs: Dict[str, BlockSpec]):
        """
        Initialize wiring with block specifications.

        Args:
            block_specs: Dictionary mapping block names to BlockSpec objects
        """
        self.block_specs = block_specs
        self.num_blocks = len(block_specs)

        # Create block name to index mapping
        self.block_names = list(block_specs.keys())
        self.name_to_idx = {name: idx for idx, name in enumerate(self.block_names)}

        # Initialize adjacency matrix (sparse connectivity)
        # Shape: [num_blocks, num_blocks]
        # A[i, j] = polarity if block i connects to block j, else 0
        # Use native Python lists instead of NumPy to avoid MLX wrapper conflicts
        self.adjacency = [[0.0 for _ in range(self.num_blocks)] for _ in range(self.num_blocks)]

        # Block instances (lazy initialization)
        self._blocks: Dict[str, nn.Module] = {}

    def add_connection(self, from_block: str, to_block: str, polarity: Optional[float] = None):
        """
        Add connection from one block to another.

        Args:
            from_block: Source block name
            to_block: Target block name
            polarity: Connection polarity (+1 excitatory, -1 inhibitory, 0 none)
                     If None, uses target block's default polarity
        """
        from_idx = self.name_to_idx[from_block]
        to_idx = self.name_to_idx[to_block]

        if polarity is None:
            polarity = self.block_specs[to_block].polarity

        self.adjacency[from_idx][to_idx] = polarity

    def remove_connection(self, from_block: str, to_block: str):
        """Remove connection between blocks"""
        from_idx = self.name_to_idx[from_block]
        to_idx = self.name_to_idx[to_block]
        self.adjacency[from_idx][to_idx] = 0.0

    def get_connections(self, block_name: str, direction: str = 'outgoing') -> List[str]:
        """
        Get connected blocks.

        Args:
            block_name: Block to query
            direction: 'outgoing' (blocks this feeds into) or 'incoming' (blocks that feed this)

        Returns:
            List of connected block names
        """
        idx = self.name_to_idx[block_name]

        if direction == 'outgoing':
            # Find non-zero columns in this row
            connected_indices = [i for i in range(self.num_blocks) if self.adjacency[idx][i] != 0]
        elif direction == 'incoming':
            # Find non-zero rows in this column
            connected_indices = [i for i in range(self.num_blocks) if self.adjacency[i][idx] != 0]
        else:
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction}")

        return [self.block_names[i] for i in connected_indices]

    def get_execution_stages(self) -> List[List[str]]:
        """
        Compute execution stages via topological sort with level assignment.

        Blocks in the same stage have no dependencies on each other and can
        execute in PARALLEL.

        Returns:
            List of stages, where each stage is a list of block names that can run in parallel

        Example:
            [[input], [head_0, head_1, head_2, head_3], [combiner], [output]]
            Stage 1: input runs alone
            Stage 2: All 4 heads run in PARALLEL
            Stage 3: combiner runs (after heads complete)
            Stage 4: output runs
        """
        # Compute in-degree for each node (count incoming connections)
        in_degree = [0] * self.num_blocks
        for j in range(self.num_blocks):
            for i in range(self.num_blocks):
                if self.adjacency[i][j] != 0:
                    in_degree[j] += 1

        stages = []
        visited = set()

        while len(visited) < self.num_blocks:
            # Find all nodes with in_degree == 0 (no unprocessed dependencies)
            current_stage = []
            for idx, name in enumerate(self.block_names):
                if idx not in visited and in_degree[idx] == 0:
                    current_stage.append(name)

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

    def get_block(self, block_name: str) -> nn.Module:
        """
        Get or instantiate a block.

        Lazy initialization: blocks are only created when first accessed.

        Args:
            block_name: Name of block to retrieve

        Returns:
            Instantiated nn.Module for this block
        """
        if block_name not in self._blocks:
            spec = self.block_specs[block_name]
            self._blocks[block_name] = self._instantiate_block(spec)

        return self._blocks[block_name]

    def _instantiate_block(self, spec: BlockSpec) -> nn.Module:
        """
        Instantiate a block from its specification.

        Args:
            spec: BlockSpec defining the block

        Returns:
            Instantiated nn.Module
        """
        # Import block classes based on type and backend
        if spec.backend == BackendType.TORCH_COMPILED:
            from ...blocks.mlstm_torch_compiled.block import (
                CompilablemLSTMBlock,
                CompilablesLSTMBlock
            )

            if spec.block_type == BlockType.MLSTM:
                return CompilablemLSTMBlock(**spec.params)
            elif spec.block_type == BlockType.SLSTM:
                return CompilablesLSTMBlock(**spec.params)
            elif spec.block_type == BlockType.FFN:
                # Implement FFN block
                d_model = spec.params.get('d_model', 512)
                ffn_dim = spec.params.get('ffn_dim', 2048)
                return nn.Sequential(
                    nn.Linear(d_model, ffn_dim),
                    nn.SiLU(),
                    nn.Linear(ffn_dim, d_model)
                )
            elif spec.block_type == BlockType.LINEAR:
                return nn.Linear(**spec.params)
            elif spec.block_type == BlockType.NORM:
                from ...blocks.mlstm_torch_compiled.block import MetalRMSNorm
                return MetalRMSNorm(spec.params.get('d_model', 512))

        elif spec.backend == BackendType.MLX:
            import mlx.nn as mlx_nn
            from ...blocks.mlstm_mlx.xlstm_block import xLSTMBlock, xLSTMBlockConfig
            from ...blocks.mlstm_mlx.components import RMSNorm

            if spec.block_type == BlockType.MLSTM:
                # Create xLSTM block (mLSTM + FFN)
                config = xLSTMBlockConfig(**spec.params)
                return xLSTMBlock(config)
            elif spec.block_type == BlockType.EMBEDDING:
                vocab_size = spec.params['vocab_size']
                embedding_dim = spec.params['embedding_dim']
                return mlx_nn.Embedding(vocab_size, embedding_dim)
            elif spec.block_type == BlockType.LINEAR:
                return mlx_nn.Linear(**spec.params)
            elif spec.block_type == BlockType.NORM:
                d_model = spec.params.get('embedding_dim') or spec.params.get('d_model')
                eps = spec.params.get('eps', 1e-6)
                force_float32 = spec.params.get('force_float32_reductions', True)
                return RMSNorm(
                    num_features=d_model,
                    eps=eps,
                    use_weight=True,
                    use_bias=False,
                    force_float32_reductions=force_float32
                )
            else:
                raise NotImplementedError(f"MLX backend not yet implemented for {spec.block_type}")

        elif spec.backend == BackendType.TRITON:
            # Triton blocks (to be implemented)
            raise NotImplementedError(f"Triton backend not yet implemented for {spec.block_type}")

        else:
            raise ValueError(f"Unknown backend: {spec.backend}")

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


class WiredMADModel(nn.Module):
    """
    MAD model using declarative wiring instead of sequential layers.

    Executes blocks according to topological ordering, with parallelism
    where dependencies allow.

    Example:
        >>> # Define architecture
        >>> specs = {...}  # BlockSpec dictionary
        >>> wiring = MADWiring(specs)
        >>> wiring.add_connection('input', 'head_0')
        >>> wiring.add_connection('input', 'head_1')
        >>> wiring.add_connection('head_0', 'combiner')
        >>> wiring.add_connection('head_1', 'combiner')
        >>>
        >>> # Create model
        >>> model = WiredMADModel(wiring, input_block='input', output_block='combiner')
        >>>
        >>> # Forward pass (heads run in parallel)
        >>> output = model(input_tensor)
    """

    def __init__(
        self,
        wiring: MADWiring,
        input_block: str,
        output_block: str,
        enable_parallel: bool = False  # TODO: Implement parallel execution
    ):
        """
        Initialize wired MAD model.

        Args:
            wiring: MADWiring defining block connectivity
            input_block: Name of block that receives input
            output_block: Name of block that produces final output
            enable_parallel: If True, run independent blocks in parallel (future)
        """
        super().__init__()

        self.wiring = wiring
        self.input_block_name = input_block
        self.output_block_name = output_block
        self.enable_parallel = enable_parallel

        # Get execution stages
        self.stages = wiring.get_execution_stages()

        # Validate input/output blocks are in graph
        if input_block not in wiring.block_names:
            raise ValueError(f"Input block '{input_block}' not in wiring")
        if output_block not in wiring.block_names:
            raise ValueError(f"Output block '{output_block}' not in wiring")

        # Instantiate all blocks and register as submodules
        self.blocks = nn.ModuleDict()
        for name in wiring.block_names:
            self.blocks[name] = wiring.get_block(name)

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass following wiring graph.

        Args:
            x: Input tensor
            hidden_states: Optional dictionary of hidden states per block

        Returns:
            output: Final output from output_block
            new_hidden_states: Updated hidden states
        """
        if hidden_states is None:
            hidden_states = {}

        # Activation cache: stores output of each block
        activations: Dict[str, torch.Tensor] = {}
        new_hidden_states: Dict[str, Any] = {}

        # Execute stages in order
        for stage_idx, stage_blocks in enumerate(self.stages):
            # TODO: Implement true parallel execution for blocks in same stage
            # For now, sequential execution within stage

            for block_name in stage_blocks:
                # Get block inputs
                incoming = self.wiring.get_connections(block_name, 'incoming')

                if not incoming:
                    # No incoming connections - this is the input block
                    block_input = x
                else:
                    # Aggregate inputs from all incoming connections
                    inputs = []
                    for source_name in incoming:
                        source_idx = self.wiring.name_to_idx[source_name]
                        target_idx = self.wiring.name_to_idx[block_name]
                        polarity = self.wiring.adjacency[source_idx][target_idx]

                        # Apply polarity scaling
                        scaled_input = polarity * activations[source_name]
                        inputs.append(scaled_input)

                    # Combine inputs (sum by default, could be configurable)
                    block_input = torch.stack(inputs).sum(dim=0)

                # Execute block
                block = self.blocks[block_name]
                block_hidden = hidden_states.get(block_name, None)

                # Block must handle hidden_states explicitly
                output, new_hidden = block(block_input, block_hidden)
                new_hidden_states[block_name] = new_hidden

                # Cache activation
                activations[block_name] = output

        # Return output block's activation
        final_output = activations[self.output_block_name]

        return final_output, new_hidden_states

    def get_execution_info(self) -> Dict[str, Any]:
        """
        Get information about execution graph.

        Returns:
            Dictionary with execution statistics
        """
        return {
            'num_blocks': self.wiring.num_blocks,
            'num_stages': len(self.stages),
            'blocks_per_stage': [len(stage) for stage in self.stages],
            'max_parallelism': max(len(stage) for stage in self.stages),
            'stages': self.stages,
            'visualization': self.wiring.visualize()
        }


def create_parallel_head_wiring(
    num_heads: int = 4,
    d_model: int = 512,
    head_dim: int = 128,
    backend: BackendType = BackendType.TORCH_COMPILED
) -> MADWiring:
    """
    Create wiring for parallel multi-head mLSTM.

    This is the canonical example from docs/MAD_WIRING_INTEGRATION.md.

    Args:
        num_heads: Number of parallel mLSTM heads
        d_model: Model dimension
        head_dim: Head dimension
        backend: Compute backend

    Returns:
        MADWiring with parallel head structure

    Example:
        >>> wiring = create_parallel_head_wiring(num_heads=4, d_model=512)
        >>> model = WiredMADModel(wiring, 'input_norm', 'output_proj')
        >>> # 4 mLSTM heads run in PARALLEL
    """
    specs = {}

    # Input normalization
    specs['input_norm'] = BlockSpec(
        name='input_norm',
        block_type=BlockType.NORM,
        backend=backend,
        params={'d_model': d_model}
    )

    # Parallel mLSTM heads
    for i in range(num_heads):
        specs[f'mlstm_head_{i}'] = BlockSpec(
            name=f'mlstm_head_{i}',
            block_type=BlockType.MLSTM,
            backend=backend,
            params={'d_model': d_model, 'num_heads': 1, 'head_dim': head_dim}
        )

    # Head combiner (concatenate + project)
    specs['output_proj'] = BlockSpec(
        name='output_proj',
        block_type=BlockType.LINEAR,
        backend=backend,
        params={'in_features': num_heads * head_dim, 'out_features': d_model}
    )

    # Create wiring
    wiring = MADWiring(specs)

    # Connect input to all heads (parallel fanout)
    for i in range(num_heads):
        wiring.add_connection('input_norm', f'mlstm_head_{i}')

    # Connect all heads to combiner (parallel fanin)
    for i in range(num_heads):
        wiring.add_connection(f'mlstm_head_{i}', 'output_proj')

    return wiring


def create_xlstm_7_1_wiring(
    d_model: int = 512,
    num_blocks: int = 8,
    backend: BackendType = BackendType.TORCH_COMPILED
) -> MADWiring:
    """
    Create wiring for 7:1 xLSTM pattern (7 mLSTM blocks, 1 sLSTM block).

    This is the canonical xLSTM architecture from the paper.

    Args:
        d_model: Model dimension
        num_blocks: Total number of blocks (should be multiple of 8)
        backend: Compute backend

    Returns:
        MADWiring with 7:1 mLSTM:sLSTM pattern
    """
    specs = {}

    # Input embedding (would be separate, just using norm for now)
    specs['input'] = BlockSpec(
        name='input',
        block_type=BlockType.NORM,
        backend=backend,
        params={'d_model': d_model}
    )

    # Create blocks in 7:1 pattern
    prev_block = 'input'
    for i in range(num_blocks):
        # Every 8th block is sLSTM, rest are mLSTM
        if (i + 1) % 8 == 0:
            block_type = BlockType.SLSTM
            block_name = f'slstm_{i}'
        else:
            block_type = BlockType.MLSTM
            block_name = f'mlstm_{i}'

        specs[block_name] = BlockSpec(
            name=block_name,
            block_type=block_type,
            backend=backend,
            params={'d_model': d_model}
        )

    # Output projection
    specs['output'] = BlockSpec(
        name='output',
        block_type=BlockType.LINEAR,
        backend=backend,
        params={'in_features': d_model, 'out_features': d_model}
    )

    # Create sequential wiring (no parallelism in this pattern)
    wiring = MADWiring(specs)

    prev_block = 'input'
    for i in range(num_blocks):
        if (i + 1) % 8 == 0:
            block_name = f'slstm_{i}'
        else:
            block_name = f'mlstm_{i}'

        wiring.add_connection(prev_block, block_name)
        prev_block = block_name

    wiring.add_connection(prev_block, 'output')

    return wiring


def create_xlstm_7b_mlx_wiring(
    embedding_dim: int = 4096,
    num_heads: int = 8,
    num_blocks: int = 32,
    vocab_size: int = 50304,
    qk_dim_factor: float = 0.5,
    v_dim_factor: float = 1.0,
    gate_soft_cap: float = 15.0,
    ffn_proj_factor: float = 2.671875,
    ffn_act_fn: str = "swish",
    norm_eps: float = 1e-6,
    output_logit_soft_cap: float = 30.0
) -> MADWiring:
    """
    Create MAD wiring for xLSTM-7B model using MLX backend.

    This creates the canonical xLSTM-7B architecture:
        embedding -> 32 xLSTM blocks -> final_norm -> lm_head

    Each xLSTM block contains:
        - Pre-norm -> mLSTM layer -> residual
        - Pre-norm -> FFN -> residual

    Args:
        embedding_dim: Model dimension (4096 for xLSTM-7B)
        num_heads: Number of attention heads (8 for xLSTM-7B)
        num_blocks: Number of xLSTM blocks (32 for xLSTM-7B)
        vocab_size: Vocabulary size (50304 for xLSTM-7B)
        qk_dim_factor: QK dimension factor (0.5 for xLSTM-7B)
        v_dim_factor: V dimension factor (1.0 for xLSTM-7B)
        gate_soft_cap: Gate soft cap value (15.0 for xLSTM-7B)
        ffn_proj_factor: FFN projection factor (2.671875 for xLSTM-7B)
        ffn_act_fn: FFN activation function
        norm_eps: Normalization epsilon
        output_logit_soft_cap: Output logit soft cap

    Returns:
        MADWiring configured for xLSTM-7B with MLX backend
    """
    specs = {}

    # Embedding layer
    specs['embedding'] = BlockSpec(
        name='embedding',
        block_type=BlockType.EMBEDDING,
        backend=BackendType.MLX,
        params={
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim
        }
    )

    # xLSTM blocks (mLSTM + FFN in each block)
    for i in range(num_blocks):
        block_name = f'xlstm_{i}'
        specs[block_name] = BlockSpec(
            name=block_name,
            block_type=BlockType.MLSTM,
            backend=BackendType.MLX,
            params={
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'qk_dim_factor': qk_dim_factor,
                'v_dim_factor': v_dim_factor,
                'gate_soft_cap': gate_soft_cap,
                'ffn_proj_factor': ffn_proj_factor,
                'ffn_act_fn': ffn_act_fn,
                'use_bias': False,
                'norm_eps': norm_eps,
                'norm_reduction_force_float32': True,
                'eps': 1e-6,
                'inference_state_dtype': 'float32',
                'return_last_states': True
            }
        )

    # Final normalization
    specs['final_norm'] = BlockSpec(
        name='final_norm',
        block_type=BlockType.NORM,
        backend=BackendType.MLX,
        params={
            'embedding_dim': embedding_dim,
            'eps': norm_eps,
            'force_float32_reductions': True
        }
    )

    # LM head (Linear layer for token prediction)
    specs['lm_head'] = BlockSpec(
        name='lm_head',
        block_type=BlockType.LINEAR,
        backend=BackendType.MLX,
        params={
            'in_features': embedding_dim,
            'out_features': vocab_size,
            'bias': False
        }
    )

    # Create sequential wiring
    wiring = MADWiring(specs)

    # Connect embedding to first block
    wiring.add_connection('embedding', 'xlstm_0')

    # Connect blocks sequentially
    for i in range(num_blocks - 1):
        wiring.add_connection(f'xlstm_{i}', f'xlstm_{i+1}')

    # Connect last block to final norm and lm head
    wiring.add_connection(f'xlstm_{num_blocks-1}', 'final_norm')
    wiring.add_connection('final_norm', 'lm_head')

    return wiring
