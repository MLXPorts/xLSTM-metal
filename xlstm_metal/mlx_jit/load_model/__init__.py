"""Model loading utilities for xLSTM-Metal."""

from .config_loader import load_config, get_mlstm_config
# from .weight_loader import load_weights_into_model  # TODO: Fix MAD references
from .safetensors_loader import load_safetensors_into_wired_model

__all__ = [
    'load_config',
    'get_mlstm_config', 
    # 'load_weights_into_model',
    'load_safetensors_into_wired_model',
]
