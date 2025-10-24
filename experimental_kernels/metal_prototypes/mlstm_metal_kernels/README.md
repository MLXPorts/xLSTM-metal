Metal kernels (current location)

Contents
- mlstm_kernels.metal: Metal shader sources (soft_cap + draft mlstm_step)
- mlstm_metal_backend.mm: ObjC++ Torch extension entry (JIT via torch.utils.cpp_extension.load)
- setup.py: Example setuptools build wiring (not required for JIT load)

Planned move
- Target location: kernels/metal/
  - shaders/mlstm_kernels.metal
  - pytorch_ext/mlstm_metal_backend.mm

Why
- Keep shader and binding code isolated from Python layer code.
- Make JIT compilation entrypoints obvious and self-contained.

Status
- soft_cap kernel is functional via ObjC++ entry.
- mlstm_step kernel is a draft; chunkwise/recurrent wiring not implemented yet.

