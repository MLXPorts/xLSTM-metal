# xLSTM on MLX (Apple Silicon)

This guide explains the MLX implementation of xLSTM in this repo: what runs on GPU, how kernels are scheduled, how streams are used for overlap, and how it contrasts with the PyTorch+MPS path coordinated by Ray.

> TL;DR: MLX gives us a single‑process, Metal‑accelerated path with optional handwritten kernels (via `mx.fast.metal_kernel`) and explicit streams for concurrency. For multi‑process coordination, dashboards, and orchestration, Ray still shines; for tight inner loops on Apple GPUs, MLX is lean and direct.

## Components

- `implementations/mlx/xlstm_mlx.py` — xLSTM in MLX; supports autoregressive decode and returning hidden state.
- `scripts/run_local_xlstm_mlx.py` — MLX runner (no Ray). Byte tokenizer by default; optional HF tokenizer.
- `mlx_fast_kernels/gemm_kernels.py` — vendored shared‑memory tiled GEMM kernels (Metal) for the final projection and other matmuls.
- `tools/mlx_streams.py` — stream‑scoped synchronization helpers (background waiters, asyncio integration, result‑triggered callbacks).

## Execution Model

- All tensors are `mlx.core` arrays; compute runs on Apple GPU via MLX.
- Prefill: embed prompt and run sequence through xLSTM blocks on a GPU stream.
- Decode: loop token‑by‑token, feeding next token and updating hidden state on the same stream.
- Optional fast output head: final projection uses a tiled Metal GEMM kernel (enable `XLSTM_MLX_FAST_HEAD=1`).

```mermaid
flowchart LR
  A[Prompt tokens] --> B[Embedding]
  B --> C[xLSTM MLX blocks]\n(mLSTM + sLSTM)
  C --> D[Sequence outputs]
  D -->|optional| E[Tiled GEMM head (Metal)]
  E --> F[Logits]
  D -->|fallback| F
  F --> G[Sample next token]
  G --> H[Loop with hidden state]
```

## Streams and Overlap

- Create a dedicated GPU stream (`s_gpu = mx.new_stream(mx.gpu)`) and run prefill/decode under `with mx.stream(s_gpu)`.
- Synchronize only at boundaries before host I/O (e.g., decode output). Use `mx.synchronize(s_gpu)` instead of global syncs.
- For host callbacks (logging, UI, checkpoints), use `tools/mlx_streams.on_stream_complete(s_gpu, cb)` so waits occur in a worker thread.
- When a result’s readiness is the trigger, use `after_eval([...], cb)` (evaluates arrays off‑thread before firing callback).

## Kernel Strategy (Metal)

- Tiled kernels created by `mx.fast.metal_kernel` with body‑only Metal source + header.
- 2D grid mapping (grid = threads; threadgroup = tile) avoids runtime integer division/mod in hot loops and follows MLX dispatch semantics.
- Shared memory tiles + coalesced loads + `fma` inner loops; cooperative unique‑writer loads remove write‑write races.
- Barriers: exactly two per K‑tile (after loads, after accumulation) with `mem_threadgroup`; every thread participates.
- Optional toggles:
  - `XLSTM_GEMM_PAD=1` — +1 padding on second tile dimension to reduce bank conflicts.
  - `XLSTM_GEMM_ALIGN_EXECW=1` — align square tile size to device `threadExecutionWidth` if allowed.
  - `XLSTM_GEMM_DB=1` — double‑buffer tiles (ping‑pong) to prefetch next tile while computing the current one.
- Tile sizes are hardware‑aware:
  - Defaults: M3 → AV(32×8), AT_B(8×32); others: 16×16.
  - Env overrides: `XLSTM_GEMM_TILE_AV="TMxT"`, `XLSTM_GEMM_TILE_ATB="TNxTK"`.
  - Runtime API: `set_gemm_tiles(av="32x8", atb="8x32")`.
  - Device tuning: optional JSON (`configs/mlx_hardware_params.json`) selects defaults per device.

## MLX vs Ray (PyTorch MPS) — When to Use What

| Dimension | MLX (this path) | Ray + PyTorch MPS (existing path) |
|---|---|---|
| Inner loop | MLX GPU ops; optional Metal kernels | `torch.compile`d step kernels on MPS |
| Scheduling | Explicit MLX streams (single process) | Ray actors, local_mode=1; CPU queue; identical math on GPU |
| Orchestration | Lightweight; no external daemons | Rich orchestration, dashboard, metrics, cluster‑ready |
| Memory | Unified memory; no D2H copies | Unified memory; MPS managed; Ray adds process overhead |
| Tuning | Tiles, fast head, stream boundaries | Chunk_size, heads_per_band, workers, Ray local_mode |
| Observability | Custom logs; add your own | Ray dashboard, xltop gauges, mem watchdog |
| Simplicity | Minimal deps; fast setup | More knobs; powerful tooling |

Rule of thumb:
- MLX for tight, single‑machine Apple GPU inference and kernel experiments.
- Ray for complex orchestration, dashboards, multi‑actor scheduling, and larger systems (still on MPS).

## CLI and Env

- Run MLX:
  ```bash
  PYTHONPATH=. XLSTM_MLX_FAST_HEAD=1 \
  conda run -n base python scripts/run_local_xlstm_mlx.py \
    --prompt "The capital of France is" --max_new_tokens 32 \
    --layers 6 --model-dim 512 --head-dim 64 --heads 8
  ```
- Override tiles (no reload):
  ```python
  from mlx_fast_kernels import gemm_kernels as gk
  gk.set_gemm_tiles(av="32x8", atb="8x32")
  ```
- Useful env:
  - `XLSTM_MLX_FAST_HEAD=1` — use tiled GEMM for final projection
  - `XLSTM_GEMM_TILE_AV`, `XLSTM_GEMM_TILE_ATB` — tile overrides

## References

- MLX Streams patterns: see `tools/mlx_streams.py`.
- Metal tiling patterns: `mlx_fast_kernels/gemm_kernels.py`.
- PyTorch+MPS design: `docs/PYTORCH_MPS_INFERENCE_ARCHITECTURE.md`.
- Numerics and dtype rules: `docs/MLX_NUMERICS_AND_DTYPE_GUIDE.md`.
