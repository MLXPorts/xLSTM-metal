"""
Auto-wiring generation from config and weights.

This module provides utilities to automatically generate NCPS wirings
from model configurations and weight dictionaries (from mx.load).
"""

from typing import Dict, List, Optional, Set, Tuple
import re

import mlx.core as mx

from .wirings import Wiring, AutoNCP, NCP


def discover_blocks(weight_keys: List[str]) -> Dict[str, Set[str]]:
    """
    Discover block structure from weight keys.
    
    Args:
        weight_keys: List of weight keys from mx.load()
        
    Returns:
        Dict mapping block names to their sub-components
        
    Example:
        {'backbone.blocks.0': {'mlstm_layer', 'ffn', 'norm_mlstm', 'norm_ffn'},
         'backbone.blocks.1': {...}, ...}
    """
    blocks = {}
    
    # Pattern: backbone.blocks.N.component.subcomponent...
    pattern = re.compile(r'^(backbone\.blocks\.\d+)\.([^.]+)')
    
    for key in weight_keys:
        match = pattern.match(key)
        if match:
            block_name = match.group(1)
            component = match.group(2)
            
            if block_name not in blocks:
                blocks[block_name] = set()
            blocks[block_name].add(component)
    
    return blocks


def discover_cells(weight_keys: List[str], component_prefix: str) -> Set[str]:
    """
    Discover cell structure within a component.
    
    Args:
        weight_keys: List of weight keys
        component_prefix: Prefix like 'backbone.blocks.0.mlstm_layer'
        
    Returns:
        Set of cell names within the component
        
    Example:
        For 'backbone.blocks.0.mlstm_layer':
        {'q', 'k', 'v', 'igate_preact', 'fgate_preact', 'ogate_preact', 
         'multihead_norm', 'out_proj'}
    """
    cells = set()
    prefix = component_prefix + '.'
    
    for key in weight_keys:
        if key.startswith(prefix):
            # Extract first level after prefix
            remainder = key[len(prefix):]
            cell_name = remainder.split('.')[0]
            cells.add(cell_name)
    
    return cells


def infer_polarity(cell_name: str) -> int:
    """
    Infer polarity from cell name.
    
    Args:
        cell_name: Name of the cell
        
    Returns:
        +1 for excitatory, -1 for inhibitory
    """
    # Forget gates are inhibitory
    if 'fgate' in cell_name.lower() or 'forget' in cell_name.lower():
        return -1
    # Everything else is excitatory
    return +1


def create_component_wiring(
    cells: List[str],
    wiring_type: str = 'auto',
    sparsity: float = 0.5,
    config: Optional[Dict] = None
) -> Wiring:
    """
    Create wiring for a component (mlstm_layer, ffn, etc.).
    
    Args:
        cells: List of cell names in the component
        wiring_type: 'auto' (AutoNCP), 'sequential', 'full', or custom
        sparsity: Sparsity level for AutoNCP (0 = fully connected, 1 = most sparse)
        config: Optional configuration dict
        
    Returns:
        Wiring object with appropriate connections
    """
    num_cells = len(cells)
    
    if wiring_type == 'auto':
        # Use AutoNCP to generate sparse wiring
        # output_size=1 means final output is produced by one cell
        wiring = AutoNCP(units=num_cells, output_size=1, sparsity_level=sparsity)
        
    elif wiring_type == 'sequential':
        # Sequential wiring: cell_0 -> cell_1 -> cell_2 -> ...
        wiring = Wiring(units=num_cells)
        for i in range(num_cells - 1):
            polarity = infer_polarity(cells[i])
            wiring.add_synapse(i, i + 1, polarity)
            
    elif wiring_type == 'full':
        # Fully connected (dense)
        wiring = Wiring(units=num_cells)
        for i in range(num_cells):
            for j in range(num_cells):
                if i != j:
                    polarity = infer_polarity(cells[i])
                    wiring.add_synapse(i, j, polarity)
    else:
        raise ValueError(f"Unknown wiring_type: {wiring_type}")
    
    return wiring


def AutoWiringFromConfig(
    weights_dict: Dict[str, mx.array],
    config: Optional[Dict] = None
) -> Tuple[Dict[str, Wiring], Dict[str, Dict[str, Wiring]]]:
    """
    Generate hierarchical wiring from weights and config.
    
    This is the main entry point for auto-wiring generation.
    
    Args:
        weights_dict: Output from mx.load() containing weight keys and arrays
        config: Optional config dict with wiring hints
        
    Returns:
        (block_wirings, component_wirings):
            - block_wirings: Dict mapping block names to their wirings
            - component_wirings: Dict[block_name][component_name] -> Wiring
            
    Example:
        weights = mx.load("model.safetensors")
        block_wirings, component_wirings = AutoWiringFromConfig(weights, config)
        
        # Access wiring for block 0's mlstm_layer
        mlstm_wiring = component_wirings['backbone.blocks.0']['mlstm_layer']
    """
    weight_keys = list(weights_dict.keys())
    
    # 1. Discover block structure
    blocks = discover_blocks(weight_keys)
    
    # 2. For each block, discover components and create wirings
    block_wirings = {}
    component_wirings = {}
    
    for block_name in sorted(blocks.keys()):
        components = sorted(blocks[block_name])
        
        # Create block-level wiring (component-to-component)
        block_wiring = Wiring(units=len(components))
        # Simple sequential: norm -> operator -> norm -> ffn
        for i in range(len(components) - 1):
            block_wiring.add_synapse(i, i + 1, +1)  # All excitatory at block level
        
        block_wirings[block_name] = block_wiring
        
        # Create component-level wirings (cell-to-cell within component)
        component_wirings[block_name] = {}
        
        for component in components:
            # Get config for this component type
            comp_config = config.get(component, {}) if config else {}
            wiring_type = comp_config.get('wiring', 'auto')
            sparsity = comp_config.get('sparsity', 0.5)
            
            # Discover cells within component
            component_prefix = f"{block_name}.{component}"
            cells = sorted(discover_cells(weight_keys, component_prefix))
            
            if not cells:
                continue
            
            # Create wiring for this component
            wiring = create_component_wiring(
                cells,
                wiring_type=wiring_type,
                sparsity=sparsity,
                config=comp_config
            )
            
            component_wirings[block_name][component] = wiring
    
    return block_wirings, component_wirings


def visualize_wiring(
    wiring: Wiring,
    cell_names: Optional[List[str]] = None
) -> str:
    """
    Create ASCII visualization of wiring.
    
    Args:
        wiring: Wiring object to visualize
        cell_names: Optional names for cells (defaults to numbers)
        
    Returns:
        String representation of the wiring graph
    """
    if cell_names is None:
        cell_names = [f"cell_{i}" for i in range(wiring.units)]
    
    lines = []
    lines.append(f"Wiring: {wiring.units} cells")
    lines.append("=" * 50)
    
    adj_matrix = wiring.adjacency_matrix
    
    for i in range(wiring.units):
        src_name = cell_names[i]
        connections = []
        
        for j in range(wiring.units):
            if adj_matrix[i, j] != 0:
                polarity = "+" if adj_matrix[i, j] > 0 else "-"
                connections.append(f"{polarity}{cell_names[j]}")
        
        if connections:
            lines.append(f"{src_name} -> {', '.join(connections)}")
    
    return "\n".join(lines)


__all__ = [
    'AutoWiringFromConfig',
    'discover_blocks',
    'discover_cells',
    'infer_polarity',
    'create_component_wiring',
    'visualize_wiring',
]
