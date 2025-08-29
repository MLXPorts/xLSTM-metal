import hashlib
import torch
from typing import Optional


def _tensor_bytes(t: torch.Tensor) -> bytes:
    if t is None:
        return b""
    return t.detach().contiguous().cpu().numpy().tobytes()


def trace_hash(x: torch.Tensor, times: Optional[torch.Tensor] = None, seed: Optional[int] = None) -> str:
    """Deterministic hash over inputs + schedule + optional seed.

    - x: (B,L,D) or (B,L,...) input features
    - times: (B,L) schedule indices or steps
    - seed: optional run seed
    Returns hex SHA256 string.
    """
    h = hashlib.sha256()
    h.update(_tensor_bytes(x))
    if times is not None:
        h.update(_tensor_bytes(times))
    if seed is not None:
        h.update(str(int(seed)).encode('utf-8'))
    return h.hexdigest()

