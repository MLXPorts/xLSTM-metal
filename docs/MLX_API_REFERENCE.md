# MLX API Reference (xLSTM)

This reference lists public MLX modules and functions in this repo.

## Model

- `implementations/mlx/xlstm_mlx.py`
  - `class xLSTM(nn.Module)`
    - `__call__(tokens: mx.array, hidden_states=None, return_hidden=False)`
      - Inputs: `tokens (B, T)` int32; optional `hidden_states` list per block
      - Returns: `logits (B, T, V)` and optionally final hidden states
    - `init_hidden(batch_size: int)` → per‑block hidden state list
  - `create_xlstm_model(...)` → `xLSTM`
    - Key args: `vocab_size`, `num_layers`, `signature=(m,s)`, `inp_dim`, `head_dim`, `head_num`, `p_factor`, `ker_size`, `dropout`

## Runner

- `scripts/run_local_xlstm_mlx.py`
  - Args: `--prompt`, `--prompt-file`, `--max_new_tokens`, `--temperature`, `--top_k`
  - Model dims: `--vocab-size`, `--layers`, `--model-dim`, `--head-dim`, `--heads`, `--signature`, `--dropout`
  - Tokenizer: `--hf-tokenizer` (else byte‑level 256‑vocab)
  - Uses a dedicated GPU stream for prefill and decode
  - Env: `XLSTM_MLX_FAST_HEAD=1` to enable tiled projection head

## Kernels (Metal)

- `mlx_fast_kernels/gemm_kernels.py`
  - `gemm_av(A: (m,n), V: (n,k)) -> (m,k)` — tiled AV GEMM
  - `gemm_at_b(A: (m,n), B: (m,k)) -> (n,k)` — tiled AᵀB GEMM
  - `set_gemm_tiles(av: str|(int,int)|None, atb: str|(int,int)|None)` — set tile sizes at runtime
  - `get_gemm_tiles() -> ((TM,T), (TN,TI,TK))` — get current tiles
  - Env overrides: `XLSTM_GEMM_TILE_AV`, `XLSTM_GEMM_TILE_ATB`
  - Advanced toggles: `XLSTM_GEMM_PAD=1` (tile padding), `XLSTM_GEMM_ALIGN_EXECW=1` (align T to execution width), `XLSTM_GEMM_DB=1` (double buffering)

- `mlx_fast_kernels/qr_kernels.py`
  - `project_coeffs(Q: (m,k), v: (m,)) -> (k,)` — c = Qᵀ v via baseline or simdgroup kernel; heuristics or env `QR_DOT_MODE`
  - `update_vector(Q: (m,k), c: (k,), v: (m,)) -> (m,)` — v − Q c using `fma`

- `mlx_fast_kernels/svd_kernels.py`
  - `power_iter_step_tiled(A: (m,n), V: (n,k)) -> (n,k)` — Z = Aᵀ (A V) using tiled GEMMs
  - `power_iter_step` — alias to tiled path (production)

- `mlx_fast_kernels/ivf_kernels.py`
  - `ivf_list_topk_l2(Q: (d,), X: (m,d), ids: (m,), k)` → `(vals: (k,), ids: (k,))`
  - Batched variant: `ivf_list_topk_l2_batch(Q: (B,d), X: (m,d), ids: (m,), k)` → `(B,k)` results
  - Device merge: `device_topk_merge(vals_parts: (P,kk), ids_parts: (P,kk), k)` → `(k,)`

## Streams

- `tools/mlx_streams.py`
  - `on_stream_complete(stream, callback, *args, executor=None, **kw)` → Thread|Future
  - `on_stream_complete_async(stream, callback, *args, loop=None, executor=None, **kw)` → awaitable
  - `after_eval(arrays, callback, *args, executor=None, **kw)` → Future

## Tuning

- `tools/mlx_tuning.py`
  - `tiles_for_gemm() -> (av: str|None, atb: str|None)` — device‑aware tile defaults via configs JSON
  - `qr_dot_mode_default() -> str` — default QR dot mode (auto|simd|simple)

- `tools/mlx_runtime.py`
  - `configure_gemm(pad: bool|None, align_execw: bool|None, double_buffer: bool|None)` — programmatic GEMM options
  - `configure_qr(dot_mode: str|None)` — programmatic QR mode
  - `configure_ivf(tpb: int|None)` — programmatic IVF threadgroup sizing
  - `get_runtime_config()` — inspect current runtime options
