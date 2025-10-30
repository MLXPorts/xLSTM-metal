#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import functools

import numpy as np
import torch

try:
    import triton.language as tl
    _torch_to_triton_dtype = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
except ImportError:
    # Triton not available
    tl = None
    _torch_to_triton_dtype = {}


def dtype2str(dtype: torch.dtype) -> str:
    """

    :param dtype:
    :return:
    """
    if dtype == torch.float32:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.float64:
        return "fp64"
    elif dtype == torch.bfloat16:
        return "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def contiguous(fn):
    """

    :param fn:
    :return:
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        """

        :param ctx:
        :param args:
        :param kwargs:
        :return:
        """
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper


def contiguous_noctx(fn):
    """

    :param fn:
    :return:
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return fn(
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper


def torch2triton_dtype(dtype):
    """

    :param dtype:
    :return:
    """
    return _torch_to_triton_dtype[dtype]


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """

    :param tensor:
    :return:
    """
    return tensor.detach().cpu().to(dtype=torch.float64).numpy()


def tensor_or_none(x):
    """

    :param x:
    :return:
    """
    return x if x is None else torch.tensor(x)

def int_or_none(x):
    """

    :param x:
    :return:
    """
    return x if x is None else int(x)