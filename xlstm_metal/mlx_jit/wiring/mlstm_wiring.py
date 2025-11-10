"""Wiring patterns for mLSTM neurons.

This module defines how mLSTM neurons are wired together in NCPS patterns.
"""

from typing import Optional
from .wirings import Wiring


class mLSTMWiring(Wiring):
    """
    Wiring pattern for multi-head mLSTM.
    
    Creates connections between mLSTM neurons (heads) based on the
    multi-head attention pattern but with recurrent gating.
    
    Args:
        num_heads: Number of mLSTM neurons (heads)
        inter_head_connections: Whether heads communicate with each other
        output_size: Number of output neurons (default: all heads output)
    """

    def __init__(
            self,
            num_heads: int,
            inter_head_connections: bool = False,
            output_size: Optional[int] = None,
    ):
        super().__init__(units=num_heads)
        self.num_heads = num_heads
        self.inter_head_connections = inter_head_connections

        if output_size is None:
            output_size = num_heads
        self.set_output_dim(output_size)

        # Setup connections
        if inter_head_connections:
            # Heads can communicate (sparse connections)
            self._add_inter_head_connections()
        # else: Independent heads (no recurrent connections between them)

    def _add_inter_head_connections(self):
        """Add sparse connections between heads."""
        # Connect adjacent heads (local connectivity)
        for i in range(self.num_heads - 1):
            # Excitatory connection to next head
            self.add_synapse(i, i + 1, polarity=1)

        # Connect every other head (skip connections)
        for i in range(0, self.num_heads - 2, 2):
            self.add_synapse(i, i + 2, polarity=1)


class mLSTMBlockWiring(Wiring):
    """
    Wiring for a complete xLSTM block (mLSTM layer + FFN).
    
    This represents the internal wiring of one xLSTM block with:
    - mLSTM neurons (multiple heads)
    - FFN neurons (up, gate, down projections)
    - Norm neurons (pre-norm for mLSTM and FFN)
    
    Args:
        num_mlstm_heads: Number of mLSTM heads
        ffn_hidden_size: FFN hidden dimension
        mlstm_inter_head: Whether mLSTM heads communicate
    """

    def __init__(
            self,
            num_mlstm_heads: int,
            ffn_hidden_size: int,
            mlstm_inter_head: bool = False,
    ):
        # Total neurons: mlstm_heads + 2 norms + ffn(up+gate+down)
        # Simplified: just mlstm + ffn for now
        total_units = num_mlstm_heads + 3  # mlstm heads + ffn(up,gate,down)
        super().__init__(units=total_units)

        self.num_mlstm_heads = num_mlstm_heads
        self.ffn_start_idx = num_mlstm_heads

        # Setup mLSTM head connections
        if mlstm_inter_head:
            for i in range(num_mlstm_heads - 1):
                self.add_synapse(i, i + 1, polarity=1)

        # Connect mLSTM output to FFN input
        # All mLSTM heads feed into FFN up projection
        ffn_up_idx = self.ffn_start_idx
        for i in range(num_mlstm_heads):
            self.add_synapse(i, ffn_up_idx, polarity=1)

        # FFN internal: up+gate → down
        ffn_gate_idx = self.ffn_start_idx + 1
        ffn_down_idx = self.ffn_start_idx + 2
        self.add_synapse(ffn_up_idx, ffn_down_idx, polarity=1)
        self.add_synapse(ffn_gate_idx, ffn_down_idx, polarity=1)

        # Output comes from FFN down projection
        self.set_output_dim(1)


class xLSTMStackWiring(Wiring):
    """
    Wiring for a stack of xLSTM blocks.
    
    This creates a deep network by stacking multiple xLSTM blocks
    with residual connections.
    
    Args:
        num_blocks: Number of xLSTM blocks to stack
        num_heads_per_block: Number of mLSTM heads per block
        residual_connections: Whether to add skip connections
    """

    def __init__(
            self,
            num_blocks: int,
            num_heads_per_block: int,
            residual_connections: bool = True,
    ):
        # Simplified: each block represented by its output neuron
        units = num_blocks
        super().__init__(units=units)

        self.num_blocks = num_blocks
        self.residual_connections = residual_connections

        # Sequential connections between blocks
        for i in range(num_blocks - 1):
            self.add_synapse(i, i + 1, polarity=1)

        # Residual connections (skip connections)
        if residual_connections and num_blocks > 2:
            for i in range(num_blocks - 2):
                # Skip one block
                self.add_synapse(i, i + 2, polarity=1)

        # Final output from last block
        self.set_output_dim(1)


__all__ = ['mLSTMWiring', 'mLSTMBlockWiring', 'xLSTMStackWiring']
