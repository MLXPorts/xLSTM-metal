"""
xLSTM Model Wiring for MLX

Helper functions to create xLSTM MAD wiring from config dict.
"""

from typing import Dict, Any

from xlstm_metal.wiring.core import MADWiring, BlockSpec, BlockType, BackendType


def create_xlstm_wiring(config: Dict[str, Any]) -> MADWiring:
    """
    Create MAD wiring for xLSTM model using MLX backend (config-driven).

    This creates the canonical xLSTM architecture:
        embedding -> N xLSTM blocks -> final_norm -> lm_head

    Each xLSTM block contains:
        - Pre-norm -> mLSTM layer -> residual
        - Pre-norm -> FFN -> residual

    Args:
        config: Configuration dict from load_config() or config.json
            Required keys:
                - embedding_dim: Model dimension
                - num_heads: Number of attention heads
                - num_blocks: Number of xLSTM blocks
                - vocab_size: Vocabulary size
                - qk_dim_factor: QK dimension factor
                - v_dim_factor: V dimension factor
                - gate_soft_cap: Gate soft cap value
                - ffn_proj_factor: FFN projection factor
                - norm_eps: Normalization epsilon
                - output_logit_soft_cap: Output logit soft cap
            Optional keys:
                - ffn_act_fn: FFN activation function (default: "swish")
                - use_bias: Whether to use bias (default: False)
                - eps: Small epsilon for numerical stability (default: 1e-6)
                - inference_state_dtype: State dtype (default: "float32")
                - return_last_states: Return states (default: True)
                - chunk_size: mLSTM chunk size (default: 64)

    Returns:
        MADWiring configured for xLSTM with MLX backend

    Example:
        >>> from xlstm_metal.utils.config_loader import load_config
        >>> config = load_config("xlstm_7b_model")
        >>> wiring = create_xlstm_wiring(config)
    """
    # Extract required parameters from config
    embedding_dim = config['embedding_dim']
    num_heads = config['num_heads']
    num_blocks = config['num_blocks']
    vocab_size = config['vocab_size']
    qk_dim_factor = config['qk_dim_factor']
    v_dim_factor = config['v_dim_factor']
    gate_soft_cap = config['gate_soft_cap']
    ffn_proj_factor = config['ffn_proj_factor']
    norm_eps = config['norm_eps']
    output_logit_soft_cap = config['output_logit_soft_cap']

    # Extract optional parameters with defaults
    ffn_act_fn = config.get('ffn_act_fn', 'swish')
    use_bias = config.get('use_bias', False)
    eps = config.get('eps', 1e-6)
    inference_state_dtype = config.get('inference_state_dtype', 'float32')
    return_last_states = config.get('return_last_states', True)
    chunk_size = config.get('chunk_size', 64)
    specs = {'embedding': BlockSpec(
        name='embedding',
        block_type=BlockType.EMBEDDING,
        backend=BackendType.MLX,
        params={
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim
        }
    )}

    # Embedding layer

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
                'use_bias': use_bias,
                'norm_eps': norm_eps,
                'norm_reduction_force_float32': True,
                'eps': eps,
                'inference_state_dtype': inference_state_dtype,
                'return_last_states': return_last_states,
                'chunk_size': chunk_size
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
        wiring.add_connection(f'xlstm_{i}', f'xlstm_{i + 1}')

    # Connect last block to output norm, then to lm head
    wiring.add_connection(f'xlstm_{num_blocks - 1}', 'out_norm')
    wiring.add_connection('out_norm', 'lm_head')

    return wiring


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

    DEPRECATED: Use create_xlstm_wiring(config) instead for config-driven approach.

    This function is kept for backwards compatibility. Prefer loading config from
    model directory and using create_xlstm_wiring().

    Args:
        embedding_dim: Model dimension (default: 4096)
        num_heads: Number of attention heads (default: 8)
        num_blocks: Number of xLSTM blocks (default: 32)
        vocab_size: Vocabulary size (default: 50304)
        qk_dim_factor: QK dimension factor (default: 0.5)
        v_dim_factor: V dimension factor (default: 1.0)
        gate_soft_cap: Gate soft cap value (default: 15.0)
        ffn_proj_factor: FFN projection factor (default: 2.671875)
        ffn_act_fn: FFN activation function (default: "swish")
        norm_eps: Normalization epsilon (default: 1e-6)
        output_logit_soft_cap: Output logit soft cap (default: 30.0)

    Returns:
        MADWiring configured for xLSTM-7B with MLX backend
    """
    # Create config dict from parameters
    config = {
        'embedding_dim': embedding_dim,
        'num_heads': num_heads,
        'num_blocks': num_blocks,
        'vocab_size': vocab_size,
        'qk_dim_factor': qk_dim_factor,
        'v_dim_factor': v_dim_factor,
        'gate_soft_cap': gate_soft_cap,
        'ffn_proj_factor': ffn_proj_factor,
        'ffn_act_fn': ffn_act_fn,
        'norm_eps': norm_eps,
        'output_logit_soft_cap': output_logit_soft_cap,
        'use_bias': False,
        'eps': 1e-6,
        'inference_state_dtype': 'float32',
        'return_last_states': True,
        'chunk_size': 64
    }

    # Delegate to config-driven function
    return create_xlstm_wiring(config)
