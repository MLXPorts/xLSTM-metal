WWDC Patterns Checklist (Applied)

Data Types

- Prefer half/fp16 for intermediates where safe; keep accumulators in float.
- Benchmark fp16 vs fp32 paths where numerics allow.

Tiling & Barriers

- 16×16 default tiles; test 32×8 and 8×32; also 16×8 and 8×16.
- Two barriers per tile iteration: after loads, after accumulate (see gemm_kernels.py).

Control Flow & Divergence

- Use branchless guards (ternary) in kernels; avoid warp divergence.

Memory Access

- Coalesced loads/stores; avoid dynamically indexed stack arrays in MSL.
- Stage tiles in threadgroup memory; reuse across FMAs.

Kernel Organization

- Avoid global barriers; split phases (e.g., QR dot then update; A@V then Aᵀ@B).
- Amortize launch overhead by doing enough work per threadgroup.

MLX Integration

- Reuse compiled kernels and pass small shape/flags buffers to avoid recompiles.
- Prefer runtime flags over JIT templating.

Device Awareness

- Query threadExecutionWidth and device name; align tile choices where helpful.
- Use tools/mlx_tuning.py to pick default tiles per device; override via set_gemm_tiles.

Bench Hooks

- lab/gemm_tile_bench.py sweeps tiles.
- lab/mlx_softcap_bench.py checks elementwise kernel wins.
- lab/mlx_sequence_precompute_scan_demo.py verifies dispatch reduction.

