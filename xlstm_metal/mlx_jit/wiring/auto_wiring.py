"""Automatic wiring generation from safetensors structure.

This module creates NCPS-compatible wiring automatically from model structure
detected in safetensors files, making the system agnostic to specific model types.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .wirings import Wiring


def analyze_safetensors_structure(model_dir: str) -> Dict[str, any]:
    """
    Analyze safetensors index to discover model structure.
    
    Args:
        model_dir: Path to model directory with model.safetensors.index.json
        
    Returns:
        Dict with model structure information:
        {
            'num_blocks': int,
            'block_components': {block_idx: [component_names]},
            'has_embedding': bool,
            'has_lm_head': bool,
            'has_out_norm': bool
        }
    """
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    # Analyze structure
    block_structure = defaultdict(set)
    has_embedding = False
    has_lm_head = False
    has_out_norm = False

    for key in index['weight_map'].keys():
        if key.startswith('backbone.blocks.'):
            parts = key.split('.')
            if len(parts) >= 4:
                block_idx = parts[2]
                component = parts[3]
                block_structure[block_idx].add(component)
        elif key.startswith('backbone.embeddings'):
            has_embedding = True
        elif key.startswith('lm_head'):
            has_lm_head = True
        elif key.startswith('backbone.out_norm'):
            has_out_norm = True

    # Convert sets to sorted lists
    block_components = {
        idx: sorted(list(comps))
        for idx, comps in sorted(block_structure.items(), key=lambda x: int(x[0]))
    }

    return {
        'num_blocks': len(block_structure),
        'block_components': block_components,
        'has_embedding': has_embedding,
        'has_lm_head': has_lm_head,
        'has_out_norm': has_out_norm,
    }


def detect_block_type(components: List[str]) -> str:
    """
    Detect block type from component names.
    
    Args:
        components: List of component names in block
        
    Returns:
        Block type string: 'mlstm', 'slstm', 'attention', etc.
    """
    components_set = set(components)

    if 'mlstm_layer' in components_set:
        return 'mlstm'
    elif 'slstm_layer' in components_set:
        return 'slstm'
    elif 'attn_layer' in components_set or 'attention' in components_set:
        return 'attention'
    else:
        return 'unknown'


class AutoWiring(Wiring):
    """
    Automatic wiring generated from model safetensors structure.
    
    Creates appropriate cell connections based on discovered model architecture.
    Follows NCPS patterns where each block is a neuron and blocks connect sequentially.
    """

    def __init__(
            self,
            model_dir: str,
            config: Optional[Dict] = None,
    ):
        """
        Create automatic wiring from model directory.
        
        Args:
            model_dir: Path to model directory with safetensors
            config: Optional config dict (loaded from config.json)
        """
        # Analyze structure
        self.structure = analyze_safetensors_structure(model_dir)
        self.config = config or {}
        self.model_dir = Path(model_dir)

        # Detect block types
        self.block_types = {}
        for idx, components in self.structure['block_components'].items():
            self.block_types[idx] = detect_block_type(components)

        # Number of units = number of blocks + special layers (embed, norm, lm_head)
        num_blocks = self.structure['num_blocks']
        num_special = sum([
            self.structure['has_embedding'],
            self.structure['has_out_norm'],
            self.structure['has_lm_head']
        ])
        units = num_blocks + num_special

        super().__init__(units=units)

        # Build connectivity
        self._build_connections()

    def _build_connections(self):
        """Build sequential connections between blocks."""
        # Sequential connections between all blocks
        num_blocks = self.structure['num_blocks']

        # Connect blocks sequentially
        for i in range(num_blocks - 1):
            self.add_synapse(i, i + 1, polarity=1)

        # Output dimension is determined by model structure
        # For xLSTM models, output comes from last block
        self.set_output_dim(1)

    def get_block_info(self, block_idx: int) -> Dict[str, any]:
        """
        Get information about a specific block.
        
        Args:
            block_idx: Block index (0 to num_blocks-1)
            
        Returns:
            Dict with block information
        """
        block_idx_str = str(block_idx)

        return {
            'index': block_idx,
            'type': self.block_types.get(block_idx_str, 'unknown'),
            'components': self.structure['block_components'].get(block_idx_str, []),
        }

    def create_block_cell(self, block_idx: int):
        """
        Create appropriate cell for this block based on detected type.
        
        Args:
            block_idx: Block index
            
        Returns:
            Appropriate cell instance (xLSTM7BCell, etc.)
        """
        block_info = self.get_block_info(block_idx)
        block_type = block_info['type']

        if block_type == 'mlstm':
            from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_block import mLSTMBlock
            return mLSTMBlock.from_config(block_idx, self.config)
        elif block_type == 'slstm':
            from xlstm_metal.mlx_jit.blocks.slstm.slstm_block import sLSTMBlock
            return sLSTMBlock.from_config(block_idx, self.config)
        elif block_type == 'attention':
            # TODO: Implement attention cell
            raise NotImplementedError("Attention cells not yet implemented")
        else:
            raise ValueError(f"Unknown block type: {block_type}")


def create_auto_wiring(model_dir: str, config: Optional[Dict] = None) -> AutoWiring:
    """
    Create automatic wiring from model directory.
    
    This is the main entry point for automatic model loading.
    
    Args:
        model_dir: Path to model directory with safetensors and config.json
        config: Optional config dict (will load from config.json if not provided)
        
    Returns:
        AutoWiring instance
        
    Example:
        >>> wiring = create_auto_wiring("xlstm_7b_model")
        >>> print(f"Detected {wiring.structure['num_blocks']} blocks")
        >>> # Use wiring to create model...
    """
    # Load config if not provided
    if config is None:
        from xlstm_metal.mlx_jit.load_model import load_config
        config = load_config(model_dir)

    return AutoWiring(model_dir, config)


__all__ = [
    'AutoWiring',
    'create_auto_wiring',
    'analyze_safetensors_structure',
    'detect_block_type',
]
