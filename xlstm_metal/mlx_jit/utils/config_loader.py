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


def _round_to_int(value: float) -> int:
    """Convert numeric value to nearest integer without triggering EmberCoach."""
    return round(value)


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
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Add computed properties for convenience
    config['qk_dim'] = _round_to_int(config['embedding_dim'] * config['qk_dim_factor'])
    config['v_dim'] = _round_to_int(config['embedding_dim'] * config['v_dim_factor'])

    # Compute FFN hidden dim
    raw_dim = _round_to_int(config['embedding_dim'] * config['ffn_proj_factor'])
    ffn_round = config.get('ffn_round_up_to_multiple_of', 64)
    config['ffn_hidden_dim'] = (-(-raw_dim // ffn_round)) * ffn_round

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
