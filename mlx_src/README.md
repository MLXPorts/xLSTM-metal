MLX World (Apple MLX / Metal)

Purpose
- Single place to group all MLX/Metal‑centric code: kernels, reference impls, tools.

Initial contents (mapping)
- mlx_fast_kernels/ (currently lives at repo root)
- mlx_implementation/ (third‑party/reference)
- implementations/mlx/ (curated in‑repo reference)

Notes
- For now, code remains at original paths; this folder establishes the
  separation and documents the mapping. New code should target this namespace.
- We will migrate files here with compatibility shims to avoid breaking imports.

