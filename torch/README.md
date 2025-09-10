Torch World (PyTorch)

Purpose
- Single place to group all PyTorch‑centric code: kernels, reference impls, tools.

Initial contents (mapping)
- mlstm_kernels/ (currently lives at repo root)
- pytorch_implementation/ (third‑party/reference)
- implementations/pytorch/ (curated in‑repo reference)

Notes
- For now, code remains at original paths; this folder establishes the
  separation and documents the mapping. New code should target this namespace.
- We will migrate files here with compatibility shims to avoid breaking imports.

