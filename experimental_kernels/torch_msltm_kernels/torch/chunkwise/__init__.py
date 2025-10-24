#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native import mlstm_chunkwise__native_autograd, mlstm_chunkwise__native_custbw

# Try to import Metal kernels
try:
    from .metal import mlstm_chunkwise__metal_autograd, mlstm_chunkwise__metal_custbw
    has_metal = True
except ImportError:
    has_metal = False

try:
    from .triton_limit_chunk import mlstm_chunkwise__limit_chunk
    from .triton_xl_chunk import mlstm_chunkwise__xl_chunk
    registry = {
        "native_autograd": mlstm_chunkwise__native_autograd,
        "native_custbw": mlstm_chunkwise__native_custbw,
        "triton_limit_chunk": mlstm_chunkwise__limit_chunk,
        "triton_xl_chunk": mlstm_chunkwise__xl_chunk,
    }
except ImportError:
    registry = {
        "native_autograd": mlstm_chunkwise__native_autograd,
        "native_custbw": mlstm_chunkwise__native_custbw,
    }

# Add Metal kernels if available (strict, Metal-only)
if has_metal:
    registry["metal_autograd"] = mlstm_chunkwise__metal_autograd
    registry["metal_custbw"] = mlstm_chunkwise__metal_custbw

# Register compiled native chunkwise variants (Triton port replacement)
try:
    from .native_compiled.fw import (
        mlstm_chunkwise__native_compiled_autograd,
        mlstm_chunkwise__native_compiled_xl,
    )
    registry["native_compiled_autograd"] = mlstm_chunkwise__native_compiled_autograd
    registry["native_compiled_xl"] = mlstm_chunkwise__native_compiled_xl
except Exception:
    # Strict: if these fail to import/compile, keep registry without them
    pass

# Add queued compiled-steps variant (GPU-only, MPS)
try:
    from .queued_compiled.driver import (
        mlstm_chunkwise__queued_compiled_steps,
    )
    registry["queued_compiled_steps"] = mlstm_chunkwise__queued_compiled_steps
except Exception:
    pass

# Add Ray actor-based compiled-steps variant (GPU-only, local_mode recommended)
try:
    from .ray_compiled.driver import (
        mlstm_chunkwise__ray_compiled_steps,
    )
    registry["ray_compiled_steps"] = mlstm_chunkwise__ray_compiled_steps
except Exception:
    pass
