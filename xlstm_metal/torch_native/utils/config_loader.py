#!/usr/bin/env python
"""
Configuration loader for xLSTM models (PyTorch backend).

Loads model configuration from HuggingFace config.json.
PyTorch uses plain dicts for configuration (not dataclasses/objects).
PyTorch transformers uses xLSTMConfig objects.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict


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

    # Defaults / runtime knobs
    config.setdefault('chunk_size', 64)
    config.setdefault('mode', 'inference')
    config.setdefault('return_last_states', True)
    config.setdefault('autocast_kernel_dtype', 'bfloat16')
    config.setdefault('inference_state_dtype', 'float32')
    config.setdefault('norm_reduction_force_float32', True)

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

__all__ = ["load_config", "get_mlstm_config"]
