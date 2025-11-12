"""
xLSTM Inference Utilities (PyTorch backend)

Configuration loading, weight loading, and checkpoint inference.
"""

from .config_loader import load_config, get_mlstm_config
from .infer_config_from_checkpoint import infer_config_from_checkpoint
from .infer_config_from_safetensors import infer_config_from_safetensors
from .safetensors_loader import load_safetensors_into_wired_model  # legacy PyTorch loader
from .weight_loader import load_npz_weights_to_block, load_weights_into_wired_model  # legacy PyTorch npz
from .dtype_utils import resolve_dtype
from .safetensors_torch_loader import load_safetensor_shards

__all__ = [
    'load_config',
    'get_mlstm_config',
    'load_safetensor_shards',
    'infer_config_from_checkpoint',
    'infer_config_from_safetensors',
    'load_safetensors_into_wired_model',
    'load_npz_weights_to_block',
    'load_weights_into_wired_model',
    'resolve_dtype',
]
