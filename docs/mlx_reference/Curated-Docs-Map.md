<!-- Note: Ported from MetalFaiss (docs/mlx). Paths and examples adapted for this xLSTM MLX project. -->

# Curated MLX Docs (External) — Reference Map

This project’s MLX + Metal kernel patterns are informed by a deeper body of work in the Ember ML repository. Those curated docs contain real‑world findings, pitfalls, and fixes that go beyond the official MLX documentation.

Primary curated path on your system:
- `/Volumes/emberstuff/Projects/magentic-codex/codex-cli/agent_knowledgebase/mlx/`

Suggested starting points (high‑signal):
- `mlx.core.fast.metal_kernel.md` — API and call contract
- `linalg.md` — QR/SVD tiling notes, kernel wrapper patterns, sizing heuristics
- `docs_curated/HPC16x8.md` — Limb‑based 128‑bit accumulation for robust dot/norm on GPU
- `docs_curated/COMMON_PITFALLS.md` — Real causes of “expected expression”, include collisions, and JIT churn
- `docs_curated/PORTING_FROM_PYTORCH.md` — RNG/key patterns; stateful to keyed conversion

Related Ember ML code references (for deeper dives):
- `ember_ml/backend/mlx/linearalg/qr_ops.py` — Enhanced QR kernel with diagnostics/safety
- `ember_ml/backend/mlx/linearalg/cholesky_ops.py` — Single‑thread and block‑tiled Cholesky
- `ember_ml/backend/mlx/linearalg/svd_ops.py` — Power‑iteration and tiling strategy
- `ember_ml/backend/mlx/linearalg/orthogonal_nonsquare.py` — Rectangular orthogonality + completion
- `ember_ml/backend/mlx/linearalg/eigen_ops.py` — Autoswitching patterns, HPC16x8 integration

Attribution
- The kernel patterns and HPC techniques used here are adapted from Sydney Bach’s Ember ML project (The Solace Project). Where applicable, this repo mirrors those approaches and salutes the ingenuity that made them work on real hardware.

How this repo applies the patterns
- docs/mlx_reference/Comprehensive-MLX-Metal-Guide.md — The definitive guide to writing, launching, and optimizing kernels with MLX in this project.
- docs/research/Journal.md — A log of experiments, benchmarks, and design rationale for our kernels.
- docs/mlx_reference/Kernel-Guide.md — Working kernel snippets (body+header), grid/tg selection, autoswitch ideas.
- docs/mlx_reference/Orthogonality.md — Practical left/right orthogonality and completion.
- `python/metalfaiss/faissmlx/kernels/qr_kernels.py` — Body‑only projection kernel; header for includes.
- `python/metalfaiss/faissmlx/qr.py` — Two‑pass MGS QR with optional kernel projections.
- `python/metalfaiss/faissmlx/svd.py` — MLX tiled subspace SVD; designed to slot in a kernelized Z‑step.

If you’re authoring new kernels, scan the curated docs first — they save days of guesswork by showing what actually compiles and runs fast on Apple GPUs.
