"""Device detection and management utilities."""

import torch

def get_best_device() -> str:
    """Get the best available device for computation."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_device_info() -> dict:
    """Get detailed information about the available device."""
    device = get_best_device()
    info = {
        'device': device,
        'device_name': None,
        'device_count': 1,
        'memory_allocated': 0,
        'memory_reserved': 0,
    }
    
    if device == "cuda":
        info['device_name'] = torch.cuda.get_device_name()
        info['device_count'] = torch.cuda.device_count()
        info['memory_allocated'] = torch.cuda.memory_allocated()
        info['memory_reserved'] = torch.cuda.memory_reserved()
    elif device == "mps":
        info['device_name'] = "Apple Silicon GPU"
        # MPS doesn't have memory query functions yet
    else:
        info['device_name'] = "CPU"
    
    return info

DEVICE = get_best_device()
print(f"Using device: {DEVICE}")