#!/usr/bin/env python
"""
Configuration loader for xLSTM models.

Loads model configuration from HuggingFace config.json.
MLX uses plain dicts for configuration (not dataclasses/objects).
PyTorch transformers uses xLSTMConfig objects.
"""

import json
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

def _round_up(value: int, multiple_of: int) -> int:
    """Match canonical round-up behavior (see transformers xLSTMConfig)."""
    if multiple_of <= 0:
        return value
    return ((value + multiple_of - 1) // multiple_of) * multiple_of


def load_config(model_path: str) -> Dict[str, Any]:
    """
    Load xLSTM configuration from HuggingFace model directory.

    Args:
        model_path: Path to model directory containing config.json

    Returns:
        Dictionary with model configuration

    Example:
        >>> config = load_config("xlstm_7b_model")
        >>> print(config['embedding_dim'])  # 4096
        >>> print(config['chunk_size'])  # 64
    """
    config_path = Path(model_path)
    if config_path.is_dir():
        config_path /= "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Compute derived dimensions - integer arithmetic only
    embedding_dim = config['embedding_dim']
    qk_factor = config['qk_dim_factor']
    v_factor = config['v_dim_factor']
    ffn_factor = config['ffn_proj_factor']
    
    # NOTE: mirror canonical transformers/xLSTMConfig behavior for rounding
    round_multiple = config.get('mlstm_round_up_to_multiple_of', 64)
    config['qk_dim'] = _round_up(int(embedding_dim * qk_factor), round_multiple)
    config['v_dim'] = _round_up(int(embedding_dim * v_factor), round_multiple)
    raw_dim = int(embedding_dim * ffn_factor)
    ffn_round = config.get('ffn_round_up_to_multiple_of', 64)
    config['ffn_hidden_dim'] = _round_up(raw_dim, ffn_round)
    config['qk_head_dim'] = config['qk_dim'] // config['num_heads']
    config['v_head_dim'] = config['v_dim'] // config['num_heads']

    # fill canonical defaults when missing
    config.setdefault('chunkwise_kernel', 'chunkwise--native_autograd')
    config.setdefault('sequence_kernel', 'native_sequence__native')
    config.setdefault('step_kernel', 'native')
    config.setdefault('mode', 'inference')
    config.setdefault('weight_mode', 'single')
    config.setdefault('max_inference_chunksize', 16384)
    config.setdefault('return_last_states', True)
    config.setdefault('autocast_kernel_dtype', 'bfloat16')
    config.setdefault('inference_state_dtype', 'float32')

    return config


def get_mlstm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract mLSTM block configuration from full model config.

    Args:
        config: Full model configuration dict

    Returns:
        Dict with parameters for mLSTMConfig initialization

    Example:
        >>> full_config = load_config("xlstm_7b_model")
        >>> mlstm_config = get_mlstm_config(full_config)
        >>> # Use for mLSTMConfig(**mlstm_config)
    """
    return {
        'embedding_dim': config['embedding_dim'],
        'num_heads': config['num_heads'],
        'qk_dim_factor': config['qk_dim_factor'],
        'v_dim_factor': config['v_dim_factor'],
        'gate_soft_cap': config['gate_soft_cap'],
        'use_bias': config['use_bias'],
        'norm_eps': config['norm_eps'],
        'norm_reduction_force_float32': config['norm_reduction_force_float32'],
        'eps': config['eps'],
        'inference_state_dtype': config['inference_state_dtype'],
        'return_last_states': config['return_last_states'],
        'chunk_size': config['chunk_size']
    }


def load_safetensor_shards(model_path: str, index_filename: str = "model.safetensors.index.json") -> Dict[str, mx.array]:
    """Load every safetensor shard in ``model_path`` using ``mx.load``.

    Args:
        model_path: Directory containing model.safetensors.* files
        index_filename: Name of the HuggingFace shard index (default: model.safetensors.index.json)

    Returns:
        Dict mapping tensor names -> MX arrays containing the weights.

    Raises:
        FileNotFoundError if the index file or shard files are missing.
    """

    model_dir = Path(model_path)
    index_path = model_dir / index_filename
    if not index_path.exists():
        raise FileNotFoundError(f"Safetensors index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    if not shard_files:
        raise FileNotFoundError(
            f"No shard entries listed in {index_path}."
        )

    #tensors: Dict[str, mx.array] = {}
    for shard in shard_files:
        shard_path = model_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file missing: {shard_path}")
        shard_data = mx.load(str(shard_path), return_metadata=False)
        #for name, array in shard_data.items():
        #    tensors[name] = array

    return shard_data

# PyTorch transformers uses xLSTMConfig class with the following structure:
#
# from transformers import AutoConfig
# config = AutoConfig.from_pretrained("xlstm_7b_model")
#
# This returns a xLSTMConfig object with attributes:
# - config.embedding_dim
# - config.num_heads
# - config.chunk_size
# etc.
#
# For MLX, we use plain dicts instead (more idiomatic for MLX).
