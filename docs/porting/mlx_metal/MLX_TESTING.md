# MLX xLSTM Testing (Focused)

Run only the MLX/xLSTM tests to validate correctness without pulling in legacy or unrelated suites.

## Kernel Parity

```
PYTHONPATH=. pytest -q \
  tests/test_mlx_gemm.py \
  tests/test_mlx_qr_kernels.py \
  tests/test_mlx_svd_kernels.py \
  tests/test_mlx_ivf_kernels.py
```

Criteria: max abs diff ≤ 1e‑4 for float32 comparisons.

## xLSTM Parity & Shapes

```
PYTHONPATH=. pytest -q \
  tests/test_xlstm_mlx_inference_parity.py \
  tests/test_xlstm_mlx_batch_decode.py
```

- Fast head ON vs OFF (argmax) must produce identical token sequences.
- Batch prefill/step shapes correct and states returned for each block.

## Notes
- Keep seeds fixed (`mx.random.seed`) for deterministic argmax sequences.
- For large vocab tests (32k–50k), allow tiny numeric jitter in logits; parity tests use argmax decode to compare tokens.

