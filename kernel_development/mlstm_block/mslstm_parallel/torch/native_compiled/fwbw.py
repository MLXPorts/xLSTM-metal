import torch

from kernel_development.mlstm_block.mslstm_parallel.vendored.torch_native.variation import (
    mlstm_parallel__native_autograd as _parallel_native_autograd,
)
from kernel_development.mlstm_block.mslstm_parallel.vendored.torch_native import (
    mlstm_parallel__native_stablef_autograd as _parallel_native_stablef_autograd,
)


def _parallel_native_compiled_autograd_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = True,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    # Strict device check: compiled path must run on GPU
    if q.device.type == 'cpu':
        raise RuntimeError("mLSTM parallel compiled requires GPU (MPS/CUDA); CPU not allowed.")
    # Delegate to existing native parallel implementation (pure PyTorch ops)
    return _parallel_native_autograd(
        q=q, k=k, v=v, i=i, f=f,
        c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
        return_last_states=return_last_states, eps=eps,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )


def _parallel_native_stablef_compiled_autograd_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = True,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    # Strict device check: compiled path must run on GPU
    if q.device.type == 'cpu':
        raise RuntimeError("mLSTM parallel compiled (stablef) requires GPU (MPS/CUDA); CPU not allowed.")
    # Delegate to stablef native implementation (pure PyTorch ops)
    return _parallel_native_stablef_autograd(
        q=q, k=k, v=v, i=i, f=f,
        c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
        return_last_states=return_last_states, eps=eps,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )


try:
    mlstm_parallel__native_compiled_autograd = torch.compile(_parallel_native_compiled_autograd_eager,
                                                             mode="reduce-overhead")
except Exception as e:
    raise RuntimeError(
        f"torch.compile failed for parallel native compiled kernel: {e}. No fallback allowed."
    )

try:
    mlstm_parallel__native_stablef_compiled_autograd = torch.compile(_parallel_native_stablef_compiled_autograd_eager,
                                                                     mode="reduce-overhead")
except Exception as e:
    raise RuntimeError(
        f"torch.compile failed for parallel native stablef compiled kernel: {e}. No fallback allowed."
    )
