#!/usr/bin/env python
"""
xLSTM-7B Model Wiring for MLX

Helper function to create xLSTM-7B MAD wiring.
"""

from mad.wiring.core import MADWiring, BlockSpec, BlockType, BackendType


def create_xlstm_7b_wiring(
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

    # Output normalization (backbone.out_norm in canonical model)
    specs['out_norm'] = BlockSpec(
        name='out_norm',
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
            'input_dims': embedding_dim,
            'output_dims': vocab_size,
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

    # Connect last block to output norm, then to lm head
    wiring.add_connection(f'xlstm_{num_blocks-1}', 'out_norm')
    wiring.add_connection('out_norm', 'lm_head')

    return wiring
