#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native import mlstm_parallel__native_autograd, mlstm_parallel__native_custbw
from .native_stablef import mlstm_parallel__native_stablef_autograd, mlstm_parallel__native_stablef_custbw

try:
    from .triton_limit_headdim import mlstm_parallel__limit_headdim
    registry = {
        "native_autograd": mlstm_parallel__native_autograd,
        "native_custbw": mlstm_parallel__native_custbw,
        "native_stablef_autograd": mlstm_parallel__native_stablef_autograd,
        "native_stablef_custbw": mlstm_parallel__native_stablef_custbw,
        "triton_limit_headdim": mlstm_parallel__limit_headdim,
    }
except ImportError:
    registry = {
        "native_autograd": mlstm_parallel__native_autograd,
        "native_custbw": mlstm_parallel__native_custbw,
        "native_stablef_autograd": mlstm_parallel__native_stablef_autograd,
        "native_stablef_custbw": mlstm_parallel__native_stablef_custbw,
    }

# Add compiled native parallel variants (strict torch.compile)
try:
    from .native_compiled.fwbw import (
        mlstm_parallel__native_compiled_autograd,
        mlstm_parallel__native_stablef_compiled_autograd,
    )
    registry["native_compiled_autograd"] = mlstm_parallel__native_compiled_autograd
    registry["native_stablef_compiled_autograd"] = mlstm_parallel__native_stablef_compiled_autograd
except Exception:
    pass
