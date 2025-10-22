#!/usr/bin/env python
"""
HRM+ Wiring Helpers for MLX Backend.

Provides convenience functions for creating HRM-enhanced xLSTM-7B configurations
with memory cubes, ACT halting, and neuromodulation.
"""

from typing import Dict
from ..core import MADWiring, BlockSpec, BlockType, BackendType
from xlstm_metal.blocks.mlstm_mlx.xlstm_block import xLSTMBlockConfig
from xlstm_metal.blocks.hrm_mlx.hrm_xlstm_block import HRMxLSTMConfig


def create_hrm_xlstm_7b_wiring(
    embedding_dim: int = 4096,
    num_blocks: int = 32,
    num_heads: int = 8,
    vocab_size: int = 50304,
    hrm_strategy: str = "per_segment",  # "none", "per_block", "per_segment", "post_process"
    hrm_segment_size: int = 8,  # Apply HRM every N blocks
    enable_act: bool = False,
    fuse_phase_keys: bool = True,
    cube_max_items: int = 65536,
    cube_topk: int = 8,
    k_5ht: float = 0.5,
    act_threshold: float = 0.5,
) -> MADWiring:
    """Create xLSTM-7B wiring with HRM+ memory augmentation.

    Args:
        embedding_dim: Model dimension (4096 for 7B)
        num_blocks: Number of xLSTM blocks (32 for 7B)
        num_heads: Number of attention heads (8 for 7B)
        vocab_size: Vocabulary size
        hrm_strategy: How to integrate HRM:
            - "none": Standard xLSTM without HRM
            - "per_block": Each block is HRM-enhanced
            - "per_segment": HRM gates every N blocks (default)
            - "post_process": Single HRM wrapper at end
        hrm_segment_size: Blocks per HRM segment (for per_segment strategy)
        enable_act: Whether to enable ACT halting telemetry
        fuse_phase_keys: Whether to fuse temporal phase encodings into cube keys
        cube_max_items: Maximum items in each memory cube
        cube_topk: Number of top-k retrievals per query
        k_5ht: Serotonin modulation strength
        act_threshold: ACT halting threshold

    Returns:
        MADWiring configured for HRM-enhanced xLSTM-7B

    Example:
        >>> # Per-segment HRM (memory gates every 8 blocks)
        >>> wiring = create_hrm_xlstm_7b_wiring(
        ...     hrm_strategy="per_segment",
        ...     hrm_segment_size=8,
        ...     enable_act=True
        ... )
        >>> model = WiredMADModel(wiring, 'embedding', 'lm_head')
    """
    specs: Dict[str, BlockSpec] = {}

    # Embedding
    specs['embedding'] = BlockSpec(
        name='embedding',
        block_type=BlockType.EMBEDDING,
        backend=BackendType.MLX,
        params={'vocab_size': vocab_size, 'embedding_dim': embedding_dim}
    )

    # xLSTM blocks with optional HRM enhancement
    if hrm_strategy == "none":
        # Standard xLSTM blocks without HRM
        for i in range(num_blocks):
            xlstm_config = xLSTMBlockConfig(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                qk_dim_factor=0.5,
                v_dim_factor=1.0,
                gate_soft_cap=15.0
            )
            specs[f'xlstm_{i}'] = BlockSpec(
                name=f'xlstm_{i}',
                block_type=BlockType.MLSTM,
                backend=BackendType.MLX,
                params={
                    'embedding_dim': embedding_dim,
                    'num_heads': num_heads,
                    'qk_dim_factor': 0.5,
                    'v_dim_factor': 1.0,
                    'gate_soft_cap': 15.0
                }
            )

    elif hrm_strategy == "per_block":
        # Each block is HRM-enhanced
        for i in range(num_blocks):
            xlstm_config = xLSTMBlockConfig(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                qk_dim_factor=0.5,
                v_dim_factor=1.0,
                gate_soft_cap=15.0
            )
            hrm_config_dict = {
                'xlstm_config': xlstm_config,
                'enable_hrm': True,
                'enable_act': enable_act,
                'fuse_phase_keys': fuse_phase_keys,
                'cube_max_items': cube_max_items,
                'cube_topk': cube_topk,
                'k_5ht': k_5ht,
                'act_threshold': act_threshold
            }
            specs[f'xlstm_{i}'] = BlockSpec(
                name=f'xlstm_{i}',
                block_type=BlockType.HRM_XLSTM,
                backend=BackendType.MLX,
                params=hrm_config_dict
            )

    elif hrm_strategy == "per_segment":
        # HRM gates applied every N blocks
        for i in range(num_blocks):
            # Standard xLSTM block
            specs[f'xlstm_{i}'] = BlockSpec(
                name=f'xlstm_{i}',
                block_type=BlockType.MLSTM,
                backend=BackendType.MLX,
                params={
                    'embedding_dim': embedding_dim,
                    'num_heads': num_heads,
                    'qk_dim_factor': 0.5,
                    'v_dim_factor': 1.0,
                    'gate_soft_cap': 15.0
                }
            )

            # Add HRM cube-gated block after every segment
            if (i + 1) % hrm_segment_size == 0:
                gate_name = f'hrm_gate_{i // hrm_segment_size}'
                specs[gate_name] = BlockSpec(
                    name=gate_name,
                    block_type=BlockType.CUBE_GATED,
                    backend=BackendType.MLX,
                    params={
                        'd_in': embedding_dim,
                        'fuse_phase_keys': fuse_phase_keys,
                        'k_5ht': k_5ht,
                        'max_items': cube_max_items,
                        'topk': cube_topk
                    }
                )

                # Optional ACT halting head
                if enable_act:
                    act_name = f'act_halt_{i // hrm_segment_size}'
                    specs[act_name] = BlockSpec(
                        name=act_name,
                        block_type=BlockType.ACT_HALTING,
                        backend=BackendType.MLX,
                        params={
                            'd_model': embedding_dim,
                            'threshold': act_threshold
                        }
                    )

    elif hrm_strategy == "post_process":
        # Standard xLSTM blocks + single HRM wrapper at end
        for i in range(num_blocks):
            specs[f'xlstm_{i}'] = BlockSpec(
                name=f'xlstm_{i}',
                block_type=BlockType.MLSTM,
                backend=BackendType.MLX,
                params={
                    'embedding_dim': embedding_dim,
                    'num_heads': num_heads,
                    'qk_dim_factor': 0.5,
                    'v_dim_factor': 1.0,
                    'gate_soft_cap': 15.0
                }
            )

        # Single HRM gate at the end
        specs['hrm_gate'] = BlockSpec(
            name='hrm_gate',
            block_type=BlockType.CUBE_GATED,
            backend=BackendType.MLX,
            params={
                'd_in': embedding_dim,
                'fuse_phase_keys': fuse_phase_keys,
                'k_5ht': k_5ht,
                'max_items': cube_max_items,
                'topk': cube_topk
            }
        )

        if enable_act:
            specs['act_halt'] = BlockSpec(
                name='act_halt',
                block_type=BlockType.ACT_HALTING,
                backend=BackendType.MLX,
                params={'d_model': embedding_dim, 'threshold': act_threshold}
            )

    # Output normalization
    specs['out_norm'] = BlockSpec(
        name='out_norm',
        block_type=BlockType.NORM,
        backend=BackendType.MLX,
        params={'embedding_dim': embedding_dim, 'eps': 1e-6, 'force_float32_reductions': True}
    )

    # LM head
    specs['lm_head'] = BlockSpec(
        name='lm_head',
        block_type=BlockType.LINEAR,
        backend=BackendType.MLX,
        params={'in_features': embedding_dim, 'out_features': vocab_size, 'bias': False}
    )

    # Create wiring and connect blocks
    wiring = MADWiring(specs)

    # Embedding -> first block
    wiring.add_connection('embedding', 'xlstm_0')

    # Connect xLSTM blocks with HRM gates based on strategy
    for i in range(num_blocks):
        curr_block = f'xlstm_{i}'
        next_block = f'xlstm_{i+1}' if i < num_blocks - 1 else None

        if hrm_strategy == "per_segment" and (i + 1) % hrm_segment_size == 0:
            # Route through HRM gate
            gate_name = f'hrm_gate_{i // hrm_segment_size}'
            wiring.add_connection(curr_block, gate_name)

            if enable_act:
                # Optional ACT monitoring (parallel observation)
                act_name = f'act_halt_{i // hrm_segment_size}'
                wiring.add_connection(gate_name, act_name)

            if next_block:
                wiring.add_connection(gate_name, next_block)
            else:
                wiring.add_connection(gate_name, 'out_norm')
        elif hrm_strategy == "post_process" and i == num_blocks - 1:
            # Last block routes through post-processing HRM
            wiring.add_connection(curr_block, 'hrm_gate')
            if enable_act:
                wiring.add_connection('hrm_gate', 'act_halt')
                wiring.add_connection('act_halt', 'out_norm')
            else:
                wiring.add_connection('hrm_gate', 'out_norm')
        else:
            # Direct connection
            if next_block:
                wiring.add_connection(curr_block, next_block)
            else:
                wiring.add_connection(curr_block, 'out_norm')

    # Final connections
    wiring.add_connection('out_norm', 'lm_head')

    return wiring
