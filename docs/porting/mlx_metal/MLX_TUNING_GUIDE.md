# MLX Tuning Guide (Apple GPU)

This guide mirrors the MPS tuning guide for the MLX path.

## Quick Wins

- Enable tiled projection head: `XLSTM_MLX_FAST_HEAD=1`
- Use a dedicated GPU stream and synchronize only at boundaries
- Try M3‑friendly tiles first: AV(32×8), AT_B(8×32); others: 16×16

## Tiles & Kernels

- Override via env:
    - `XLSTM_GEMM_TILE_AV="TMxT"` (e.g., `32x8`)
    - `XLSTM_GEMM_TILE_ATB="TNxTK"` (e.g., `8x32`)
- Override at runtime:
  ```python
  from mlx_fast_kernels import gemm_kernels as gk
  gk.set_gemm_tiles(av="32x8", atb="8x32")
  print(gk.get_gemm_tiles())
  ```
- Kernels: see `mlx_fast_kernels/gemm_kernels.py` for details (shared memory tiles, fma, 2D mapping).

Programmatic configuration (preferred)

- Use `tools/mlx_runtime.py` to configure behavior in code:
    - `configure_gemm(pad=True|False, align_execw=True|False, double_buffer=True|False)`
    - `configure_qr(dot_mode="auto|simd|simple")`
    - `configure_ivf(tpb=int)`

Env toggles (fallback)

- `XLSTM_GEMM_PAD=1`: +1 tile padding
- `XLSTM_GEMM_ALIGN_EXECW=1`: align square tile to execution width
- `XLSTM_GEMM_DB=1`: double‑buffered tiles
- (Reserved) `XLSTM_GEMM_VEC4`: vectorized load prototype gate

## Streams

- Create streams per device and keep a small, consistent set
- Use `with mx.stream(s_gpu): ...` for blocks of work
- Avoid global `mx.synchronize()`; prefer `mx.synchronize(s_gpu)` at boundaries
- For callbacks, use `tools/mlx_streams.on_stream_complete` or `after_eval`

## Sampling & Precision

- Keep accumulation in float32 in the final head if inputs are lower precision
- Consider top‑k sampling to reduce softmax cost on large vocabularies
- Byte tokenizer path (vocab 256) is a good quick functional test

## Example

```bash
PYTHONPATH=. XLSTM_MLX_FAST_HEAD=1 \
conda run -n base python scripts/run_local_xlstm_mlx.py \
  --prompt "The capital of France is" --max_new_tokens 32 \
  --layers 6 --model-dim 512 --head-dim 64 --heads 8
```
