#!/usr/bin/env python
"""
Configuration Loader – MLX Implementation (HuggingFace Config Parsing)

Overview
--------
Loads xLSTM model configuration from HuggingFace-style `config.json` files
and computes derived dimensions following canonical dimension rounding rules.

This module bridges the gap between:
  - HuggingFace checkpoint format (config.json with base hyperparameters)
  - Runtime model instantiation (needs computed dimensions)

Why Separate Config Loading?
-----------------------------
Model checkpoints store **base parameters** (embedding_dim, qk_dim_factor)
but runtime code needs **computed dimensions** (qk_dim, qk_head_dim). This
loader:
  1. Reads config.json
  2. Computes derived dimensions with proper rounding
  3. Fills missing defaults for inference mode
  4. Returns a complete runtime configuration dict

Dimension Computation
---------------------
Given base parameters:
  - embedding_dim (e.g., 4096)
  - qk_dim_factor (e.g., 0.5)
  - num_heads (e.g., 8)
  - mlstm_round_up_to_multiple_of (e.g., 64)

Derived dimensions:
  qk_dim_raw = int(embedding_dim * qk_dim_factor)  # 2048
  qk_dim = round_up(qk_dim_raw, 64)                # 2048 (already multiple)
  qk_head_dim = qk_dim // num_heads                # 256

Rounding ensures dimensions align with hardware SIMD widths and safetensors
weight shapes.

Configuration Hierarchy
-----------------------
1. **Required** (must be in config.json):
   - embedding_dim, vocab_size, num_blocks, num_heads
   - qk_dim_factor, v_dim_factor, ffn_proj_factor
   - gate_soft_cap, norm_eps, use_bias

2. **Optional with defaults**:
   - chunk_size: 64
   - autocast_kernel_dtype: "bfloat16"
   - inference_state_dtype: "float32"
   - norm_reduction_force_float32: True
   - max_inference_chunksize: 16384

3. **Computed**:
   - qk_dim, v_dim, ffn_hidden_dim
   - qk_head_dim, v_head_dim

MLX vs PyTorch Config
----------------------
- **MLX**: Uses plain dicts (config['embedding_dim'])
- **PyTorch**: Uses dataclass objects (config.embedding_dim)

This module produces MLX-style dicts. For PyTorch compat, see
torch_native config loaders.

Inference Defaults
------------------
When `mode` is not specified, defaults to "inference" which:
  - Uses bfloat16 for compute (autocast_kernel_dtype)
  - Uses float32 for state (inference_state_dtype)
  - Enables return_last_states for stateful generation
  - Sets max chunk size for memory-efficient prefill

Usage
-----
Basic loading:
  >>> config = load_config("xlstm_7b_model")
  >>> model = WiredxLSTM.from_config(config)

Extract mLSTM-specific config:
  >>> mlstm_cfg = get_mlstm_config(config)
  >>> block = mLSTMBlock(**mlstm_cfg)

Parity
------
Dimension computation mirrors transformers.xLSTMConfig for checkpoint
compatibility.
"""

import json
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

def _round_up(value: int, multiple_of: int) -> int:
    """Round value up to nearest multiple (matches HuggingFace xLSTMConfig).

    Parameters
    ----------
    value : int
        Raw dimension value.
    multiple_of : int
        Alignment boundary (typically 64 for SIMD/safetensors).

    Returns
    -------
    rounded : int
        Value rounded up to nearest multiple.

    Examples
    --------
    >>> _round_up(2048, 64)
    2048
    >>> _round_up(2050, 64)
    2112
    """
    if multiple_of <= 0:
        return value
    return ((value + multiple_of - 1) // multiple_of) * multiple_of


def load_config(model_path: str) -> Dict[str, Any]:
    """Load xLSTM configuration from HuggingFace model directory.

    Reads config.json, computes derived dimensions, fills defaults.

    Parameters
    ----------
    model_path : str
        Path to model directory containing config.json (or path to config.json directly).

    Returns
    -------
    config : dict
        Complete configuration with base + derived + default parameters.

    Raises
    ------
    FileNotFoundError
        If config.json not found in model_path.

    Examples
    --------
    >>> config = load_config("xlstm_7b_model")
    >>> config['embedding_dim']
    4096
    >>> config['qk_dim']  # computed from qk_dim_factor
    2048
    >>> config['qk_head_dim']  # computed from qk_dim / num_heads
    256
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
    """Extract mLSTM block parameters from full model config.

    Filters full config to only parameters needed for mLSTMBlock construction.

    Parameters
    ----------
    config : dict
        Full model configuration (from load_config).

    Returns
    -------
    mlstm_config : dict
        Subset of config relevant to mLSTMBlock initialization.

    Example
    -------
    >>> full_config = load_config("xlstm_7b_model")
    >>> mlstm_cfg = get_mlstm_config(full_config)
    >>> block = mLSTMBlock(**mlstm_cfg)
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
