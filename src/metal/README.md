# implementations/metal

Metal/MPS-oriented experimental implementations.

Files
- `xlstm_metal_optimized.py`, `xlstm_unified_metal.py` — MPS-optimized code paths.
- `xlstm_metal_complete.py` — Broader Metal-focused variant.
- `xlstm_metal_kernels.py` — Support kernels/utilities.
- `xlstm_metal_hpc_limb.py`, `xlstm_metal_hpc_limb_fixed.py` — Extended precision experiments.
- `xlstm_jit_metal.py`, `xlstm_jit_simple.py` — JIT/experimental variants.

Note
- Current production path uses compiled MPS step/sequence kernels; handwritten Metal not required for inference.

