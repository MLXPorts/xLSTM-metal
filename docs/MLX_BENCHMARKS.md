# MLX xLSTM Benchmarks

This runner sweeps realistic profiles and tile settings, capturing prefill and decode throughput for the MLX implementation.

## Runner

```
PYTHONPATH=. python scripts/benchmarks/run_mlx_benchmarks.py \
  --profiles medium large \
  --tiles "16x16,32x8,8x32" \
  --seq-len 2048 --new-tokens 256 --repeats 3 \
  --gemm-pad 1 --gemm-align-execw 1 --gemm-double-buffer 1 \
  --outdir runs/benchmarks/mlx --make-charts 1
```

- Profiles
  - `medium`: 16 layers, model_dim=1536, head_dim=128, heads=12, vocab=32k
  - `large`: 24 layers, model_dim=3072, head_dim=128, heads=24, vocab=50k
- Tiles: `TMxT` for AV and `(T,T)` for AT_B
- Repeats: the runner records medians across repeated runs
- Charts: bar charts of decode tokens/s per tile per profile (if matplotlib installed)

## Outputs

- CSV: `runs/benchmarks/mlx/<timestamp>/mlx_benchmarks.csv`
- PNG: `runs/benchmarks/mlx/<timestamp>/bench_decode_<profile>.png`

Columns (CSV)
- profile, layers, model_dim, head_dim, heads, vocab
- seq_len, new_tokens, tiles_av, tiles_atb, fast_head
- prefill_s, prefill_tok_s, decode_s, decode_tok_s, total_s

## Tips
- Use `tools/mlx_runtime.configure_model(fast_head=True)` to ensure the tiled GEMM head is active for large vocabularies.
- Compare decode throughput with/without `pad`, `align_execw`, and `double_buffer` toggles.
- Keep batch size B=1 for decode unless you’re measuring batched generation.

