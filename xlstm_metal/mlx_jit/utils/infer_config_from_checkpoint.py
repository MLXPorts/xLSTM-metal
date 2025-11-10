#!/usr/bin/env python
"""
Checkpoint-agnostic config inference dispatcher.

Tries to infer config from checkpoint (safetensors/GGUF) first,
falls back to config.json with warning.
"""

import warnings
from pathlib import Path
from typing import Dict, Any

from .config_loader import load_config
from .gguf_loader import infer_config_from_gguf
from .infer_config_from_safetensors import infer_config_from_safetensors


def infer_config_from_checkpoint(model_path: str) -> Dict[str, Any]:
    """
    Infer model configuration from checkpoint, with fallbacks.

    Priority order:
    1. Safetensors (if model.safetensors.index.json exists)
    2. GGUF (if .gguf file exists)
    3. config.json (with warning about non-model-agnostic loading)

    Args:
        model_path: Path to model directory

    Returns:
        Dict with model configuration

    Raises:
        FileNotFoundError: If no valid checkpoint or config found
    """
    p = Path(model_path)

    # Try safetensors first
    if (p / "model.safetensors.index.json").exists():
        print("Inferring config from safetensors checkpoint...")
        return infer_config_from_safetensors(model_path)

    # Try GGUF
    gguf_files = list(p.glob("*.gguf"))
    if gguf_files:
        print("Inferring config from GGUF checkpoint...")
        return infer_config_from_gguf(str(gguf_files[0]))

    # Fallback to config.json with warning
    config_path = p / "config.json"
    if config_path.exists():
        warnings.warn(
            f"No checkpoint found in {model_path}. "
            "Falling back to config.json - this is not model-agnostic! "
            "Consider using safetensors format for proper model-agnostic loading.",
            UserWarning,
            stacklevel=2
        )
        print("Loading config from config.json (not model-agnostic)...")
        return load_config(model_path)

    raise FileNotFoundError(
        f"No checkpoint or config found in {model_path}. "
        "Expected model.safetensors.index.json, *.gguf, or config.json"
    )
