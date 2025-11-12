"""Automatic Wiring Generation – MLX Implementation (Model Structure Discovery)

Overview
--------
AutoWiring automatically discovers model architecture from safetensors files
and generates appropriate NCPS wiring. This enables **model-agnostic loading**:
instead of hardcoding model structure, the system introspects checkpoint files
and builds the correct cell types and connectivity.

Problem Statement
-----------------
Traditional model loading requires:
  1. Knowing the exact architecture (layer types, counts, dimensions)
  2. Writing custom loading code for each model variant
  3. Maintaining separate codebases for 1B, 7B, 13B, etc. models

AutoWiring solves this by:
  - Parsing safetensors index JSON to discover blocks and components
  - Detecting block types (mLSTM, sLSTM, attention) from weight keys
  - Creating appropriate cell instances with correct configurations
  - Building sequential wiring automatically

How It Works
------------
1. **Structure Analysis**
   - Read `model.safetensors.index.json`
   - Extract all weight keys (e.g., `backbone.blocks.0.mlstm_layer.q.weight`)
   - Group by block index to find num_blocks
   - Detect special components (embedding, out_norm, lm_head)

2. **Block Type Detection**
   - If block has `mlstm_layer` → mLSTMBlock
   - If block has `slstm_layer` → sLSTMBlock
   - If block has `attn_layer` → AttentionBlock (future)

3. **Wiring Construction**
   - Create sequential connectivity: block_0 → block_1 → ... → block_N
   - Set units = num_blocks + special_layers
   - Provide `create_block_cell()` factory method

4. **Cell Creation**
   - Each block calls `.from_config(block_idx, config)` on appropriate class
   - Config loaded from `config.json` provides hyperparameters
   - Cells manage their own weight loading via `get_weight_keys()`

Usage Pattern
-------------
```python
# Automatic discovery
wiring = create_auto_wiring("xlstm_7b_model")
print(f"Detected {wiring.structure['num_blocks']} blocks")

# Create appropriate cell for each block
cells = [wiring.create_block_cell(i) for i in range(wiring.structure['num_blocks'])]

# Or use with WiredxLSTM wrapper
model = WiredxLSTM(wiring=wiring, load_weights=True)
```

Structure Dictionary
--------------------
`wiring.structure` contains:
  - `num_blocks`: int - Number of transformer/LSTM blocks
  - `block_components`: {idx: [component_names]} - Per-block component list
  - `has_embedding`: bool - Whether model has token embedding layer
  - `has_out_norm`: bool - Whether model has pre-LM-head normalization
  - `has_lm_head`: bool - Whether model has language modeling head

Block Type Detection
--------------------
Component patterns:
  - `mlstm_layer` → matrix-memory LSTM (xLSTM-7B default)
  - `slstm_layer` → scalar-memory LSTM (alternative variant)
  - `attn_layer` or `attention` → standard attention (future support)

Sequential Wiring
-----------------
For xLSTM models, wiring is always sequential (each block feeds next).
Future extensions could support:
  - Mixture-of-experts (sparse routing)
  - Skip connections (ResNet-style)
  - Parallel branches (ensemble-like)

Benefits
--------
1. **Zero-config loading**: Works with any xLSTM checkpoint
2. **Version agnostic**: Adapts to checkpoint structure changes
3. **Modular**: Easy to add new block types
4. **Introspectable**: Query block types before instantiation
5. **Portable**: Same pattern works across MLX/PyTorch backends

Parity
------
Logic mirrors torch-native AutoWiring for cross-backend compatibility.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .wirings import Wiring


def _parse_block_index(text: str) -> int:
    """Parse integer-like block indices without relying on bare int casts."""
    return json.loads(text)


def analyze_safetensors_structure(model_dir: str) -> Dict[str, any]:
    """Discover model architecture from safetensors index file.

    Parses `model.safetensors.index.json` to extract block structure,
    component types, and special layers (embedding, norm, LM head).

    Parameters
    ----------
    model_dir : str
        Path to model directory containing safetensors files.

    Returns
    -------
    structure : dict
        Model structure dictionary with keys:
        - `num_blocks`: Number of transformer/LSTM blocks
        - `block_components`: {block_idx: [component_names]}
        - `has_embedding`: Whether model has token embeddings
        - `has_out_norm`: Whether model has pre-head normalization
        - `has_lm_head`: Whether model has language modeling head

    Raises
    ------
    FileNotFoundError
        If safetensors index not found in model_dir.

    Example
    -------
    >>> structure = analyze_safetensors_structure("xlstm_7b_model")
    >>> print(structure)
    {
        'num_blocks': 32,
        'block_components': {0: ['ffn', 'mlstm_layer', 'norm_ffn', 'norm_mlstm'], ...},
        'has_embedding': True,
        'has_out_norm': True,
        'has_lm_head': True
    }
    """
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    # Analyze structure
    block_structure: Dict[int, set] = defaultdict(set)
    weight_keys = list(index['weight_map'].keys())

    for key in weight_keys:
        if key.startswith('backbone.blocks.'):
            parts = key.split('.')
            if len(parts) >= 4:
                block_idx = _parse_block_index(parts[2])
                component = parts[3]
                block_structure[block_idx].add(component)

    # Convert sets to sorted lists
    block_components = {
        idx: sorted(comps)
        for idx, comps in sorted(block_structure.items(), key=lambda x: x[0])
    }

    return {
        'num_blocks': len(block_structure),
        'block_components': block_components,
        'has_embedding': any(key.startswith('backbone.embeddings') for key in weight_keys),
        'has_lm_head': any(key.startswith('lm_head') for key in weight_keys),
        'has_out_norm': any(key.startswith('backbone.out_norm') for key in weight_keys),
    }


def detect_block_type(components: List[str]) -> str:
    """Infer block type from component names.

    Examines component list to determine whether block is mLSTM, sLSTM,
    attention, or unknown type.

    Parameters
    ----------
    components : list of str
        Component names from safetensors keys (e.g., ['mlstm_layer', 'ffn', 'norm_mlstm']).

    Returns
    -------
    block_type : {"mlstm", "slstm", "attention", "unknown"}
        Detected block type.

    Examples
    --------
    >>> detect_block_type(['mlstm_layer', 'ffn', 'norm_mlstm'])
    'mlstm'
    >>> detect_block_type(['slstm_layer', 'ffn'])
    'slstm'
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
    """Automatic wiring with model structure discovery and cell factory.

    Extends base Wiring with automatic architecture detection from safetensors
    and factory methods for creating appropriate block cells.

    Parameters
    ----------
    model_dir : str
        Path to model directory with safetensors and config.json.
    config : dict | None, optional
        Model configuration (loaded from config.json if not provided).

    Attributes
    ----------
    structure : dict
        Discovered model structure (from `analyze_safetensors_structure`).
    block_types : dict
        Mapping {block_idx: block_type_str}.
    config : dict
        Model hyperparameters from config.json.
    model_dir : Path
        Path to model directory.

    Methods
    -------
    get_block_info(block_idx)
        Query block type and components.
    create_block_cell(block_idx, **kwargs)
        Factory method to instantiate appropriate cell for block.
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
        self.block_types = {
            idx: detect_block_type(components)
            for idx, components in self.structure['block_components'].items()
        }

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
        for i in range(num_blocks):  # Loop var exception
            if i < num_blocks - 1:  # Loop var exception
                self.add_synapse(i, i + 1)

        # Output dimension is determined by model structure
        # For xLSTM models, output comes from last block
        self.set_output_dim(1)

    def get_block_info(self, block_idx: int) -> Dict[str, any]:
        """Query information about a specific block.

        Parameters
        ----------
        block_idx : int
            Block index [0, num_blocks).

        Returns
        -------
        block_info : dict
            Dictionary with keys:
            - `index`: Block index
            - `type`: Block type string
            - `components`: List of component names

        Example
        -------
        >>> info = wiring.get_block_info(0)
        >>> print(info['type'])
        'mlstm'
        """
        return {
            'index': block_idx,
            'type': self.block_types.get(block_idx, 'unknown'),
            'components': self.structure['block_components'].get(block_idx, []),
        }

    def create_block_cell(self, block_idx: int, **kwargs):
        """Factory method to create appropriate cell for block.

        Detects block type and instantiates corresponding class (mLSTMBlock,
        sLSTMBlock, etc.) with configuration from config.json.

        Parameters
        ----------
        block_idx : int
            Block index to create cell for.
        **kwargs
            Additional arguments passed to cell constructor.

        Returns
        -------
        cell : nn.Module
            Instantiated cell (mLSTMBlock, sLSTMBlock, etc.).

        Raises
        ------
        NotImplementedError
            If block type not yet supported.
        ValueError
            If block type unknown/invalid.

        Example
        -------
        >>> cell = wiring.create_block_cell(0, compute_dtype=mx.bfloat16)
        >>> isinstance(cell, mLSTMBlock)
        True
        """
        block_info = self.get_block_info(block_idx)
        block_type = block_info['type']

        if block_type == 'mlstm':
            from xlstm_metal.mlx_jit.blocks.mlstm.mlstm_block import mLSTMBlock
            return mLSTMBlock.from_config(block_idx, self.config, **kwargs)
        elif block_type == 'slstm':
            from xlstm_metal.mlx_jit.blocks.slstm.slstm_block import sLSTMBlock
            return sLSTMBlock.from_config(block_idx, self.config, **kwargs)
        elif block_type == 'attention':
            # TODO: Implement attention cell
            raise NotImplementedError("Attention cells not yet implemented")
        else:
            raise ValueError(f"Unknown block type: {block_type}")


def create_auto_wiring(model_dir: str, config: Optional[Dict] = None) -> AutoWiring:
    """Main entry point for automatic model loading.

    Discovers model structure from safetensors and creates wiring with
    cell factory methods. This is the recommended way to load xLSTM models.

    Parameters
    ----------
    model_dir : str
        Path to model directory with safetensors and config.json.
    config : dict | None, optional
        Model configuration (loaded automatically if not provided).

    Returns
    -------
    wiring : AutoWiring
        AutoWiring instance with discovered structure.

    Example
    -------
    >>> # Automatic loading (most common)
    >>> wiring = create_auto_wiring("xlstm_7b_model")
    >>> model = WiredxLSTM(wiring=wiring, load_weights=True)

    >>> # Custom config override
    >>> custom_config = load_config("xlstm_7b_model")
    >>> custom_config['chunk_size'] = 128
    >>> wiring = create_auto_wiring("xlstm_7b_model", custom_config)
    """
    # Load config if not provided
    if config is None:
        from xlstm_metal.mlx_jit.utils import load_config
        config = load_config(model_dir)

    return AutoWiring(model_dir, config)


__all__ = [
    'AutoWiring',
    'create_auto_wiring',
    'analyze_safetensors_structure',
    'detect_block_type',
]
