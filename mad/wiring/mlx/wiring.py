#!/usr/bin/env python
"""
MAD Wiring for MLX Backend

MLX-specific wiring system with WiredMADModel using mlx.nn.Module.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from mad.wiring.core import MADWiring, BlockSpec, BlockType, BackendType


class WiredMADModel(nn.Module):
    """
    MAD model using declarative wiring for MLX backend.

    Executes blocks according to topological ordering, with parallelism
    where dependencies allow.

    Example:
        >>> from mad.wiring.mlx import create_xlstm_7b_wiring, WiredMADModel
        >>> wiring = create_xlstm_7b_wiring()
        >>> model = WiredMADModel(wiring, 'embedding', 'lm_head')
        >>> logits, state = model(input_ids)
    """

    def __init__(
        self,
        wiring: MADWiring,
        input_block: str,
        output_block: str,
        debug: bool = False
    ):
        """
        Initialize wired MAD model for MLX.

        Args:
            wiring: MADWiring defining block connectivity
            input_block: Name of block that receives input
            output_block: Name of block that produces final output
            debug: If True, print debug info during forward pass
        """
        super().__init__()

        self.wiring = wiring
        self.input_block_name = input_block
        self.output_block_name = output_block
        self.debug = debug

        # Get execution stages
        self.stages = wiring.get_execution_stages()

        if self.debug:
            print(f"\n[MAD DEBUG] Wiring initialized:")
            print(f"  Total blocks: {len(wiring.block_names)}")
            print(f"  Execution stages: {len(self.stages)}")
            print(f"  Input block: {input_block}")
            print(f"  Output block: {output_block}")
            print(f"  Stages: {self.stages}")

        # Validate input/output blocks are in graph
        if input_block not in wiring.block_names:
            raise ValueError(f"Input block '{input_block}' not in wiring")
        if output_block not in wiring.block_names:
            raise ValueError(f"Output block '{output_block}' not in wiring")

        # Instantiate all blocks
        self._blocks = {}
        for name in wiring.block_names:
            self._blocks[name] = self._instantiate_block(wiring.block_specs[name])

    def _instantiate_block(self, spec: BlockSpec):
        """
        Instantiate a block from its specification (MLX backend only).

        Args:
            spec: BlockSpec defining the block

        Returns:
            Instantiated block (nn.Module for neural blocks, other for tokenizer)
        """
        if spec.backend != BackendType.MLX:
            raise ValueError(f"WiredMADModel (MLX) can only instantiate MLX blocks, got {spec.backend}")

        from mad.blocks.mlstm_mlx.xlstm_block import xLSTMBlock, xLSTMBlockConfig
        from mad.blocks.mlstm_mlx.components import RMSNorm
        from mad.blocks.tokenizer.block import TokenizerBlock, TokenizerConfig

        if spec.block_type == BlockType.TOKENIZER:
            # Tokenizer is not an nn.Module
            config = TokenizerConfig(**spec.params)
            return TokenizerBlock(config)
        elif spec.block_type == BlockType.MLSTM:
            # Create xLSTM block (mLSTM + FFN)
            config = xLSTMBlockConfig(**spec.params)
            return xLSTMBlock(config)
        elif spec.block_type == BlockType.EMBEDDING:
            vocab_size = spec.params['vocab_size']
            embedding_dim = spec.params['embedding_dim']
            return nn.Embedding(vocab_size, embedding_dim)
        elif spec.block_type == BlockType.LINEAR:
            return nn.Linear(**spec.params)
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

    @property
    def blocks(self) -> Dict[str, nn.Module]:
        """Access to blocks dict for weight loading."""
        return self._blocks

    def __call__(
        self,
        x: mx.array,
        hidden_states: Optional[Dict[str, Any]] = None
    ) -> Tuple[mx.array, Dict[str, Any]]:
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
        activations: Dict[str, mx.array] = {}
        new_hidden_states: Dict[str, Any] = {}

        # Execute stages in order
        for stage_idx, stage_blocks in enumerate(self.stages):
            # Sequential execution within stage (parallel TODO)

            if self.debug:
                print(f"\n[MAD DEBUG] Executing stage {stage_idx}: {stage_blocks}")

            for block_name in stage_blocks:
                # Get block inputs
                incoming = self.wiring.get_connections(block_name, 'incoming')

                if not incoming:
                    # No incoming connections - this is the input block
                    block_input = x
                    if self.debug:
                        print(f"  [{block_name}] Input from user: shape={block_input.shape}")
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

                    # Combine inputs (sum)
                    block_input = mx.stack(inputs).sum(axis=0)
                    if self.debug:
                        print(f"  [{block_name}] Input from {incoming}: shape={block_input.shape}")

                # Execute block
                block = self._blocks[block_name]
                block_hidden = hidden_states.get(block_name, None)

                # Check if block returns state tuple (like xLSTMBlock)
                output = block(block_input, block_hidden) if block_hidden is not None else block(block_input)

                # Handle state returns
                if isinstance(output, tuple) and len(output) == 2:
                    output, new_hidden = output
                    new_hidden_states[block_name] = new_hidden
                    if self.debug:
                        print(f"  [{block_name}] Output: shape={output.shape}, has_state=True")
                elif block_hidden is not None:
                    # Block expected state but didn't return it
                    if self.debug:
                        print(f"  [{block_name}] Output: shape={output.shape}, expected_state but got none")
                else:
                    if self.debug:
                        print(f"  [{block_name}] Output: shape={output.shape}, stateless")

                # Cache activation
                activations[block_name] = output

        # Return output block's activation
        final_output = activations[self.output_block_name]

        return final_output, new_hidden_states

    def get_execution_info(self) -> Dict[str, Any]:
        """
        Get information about execution graph.

        Returns:
            Dictionary with execution info
        """
        return {
            'num_blocks': len(self._blocks),
            'num_stages': len(self.stages),
            'max_parallelism': max(len(stage) for stage in self.stages),
            'block_names': list(self._blocks.keys()),
            'input_block': self.input_block_name,
            'output_block': self.output_block_name
        }
