# Torch vs MLX Numeric Stability Findings

## Context

During the Hyena/Monarch mixer port from PyTorch to MLX we audited every arithmetic path that impacts long-convolution
accuracy and determinism. The goal was bit-level parity between the Torch reference implementation (
`bert/src/mm/hyena_utils.py`) and the MLX implementation (`bert/src/mm_mlx/hyena_filter_mlx.py`). This document captures
the discrepancies we found, why they happen, and the guardrails we put in place.

## Summary of Key Differences

| Area                       | PyTorch Behaviour                                                                                        | MLX Behaviour                                                                         | Impact                                                                                                                            |
|----------------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **FFT normalisation**      | `irfft(..., norm='forward')` (no implicit scale) while manually dividing the kernel spectrum by `n = 2L` | `irfft(...)` applies `1/n` by default and we *must not* pre-divide the spectrum       | Double scaling or missing scaling gives order-one amplitude errors; fixed by matching exactly one `1/n` in each profile           |
| **Kernel combine**         | Time-domain pad + reverse + sum (default in the repo)                                                    | Selectable: time-domain sum (`torch_like`) or frequency-domain average (`mlx_stable`) | Different combine domains re-order additions; parity tests now switch via HyperProfile                                            |
| **Linear / depthwise ops** | GEMM / grouped conv implementations differ in accumulation order + FMA handling                          | Same issue on MLX (different kernels)                                                 | ~1e-7 relative drift purely from order-of-operations; acceptable but highlighted by tracer                                        |
| **float()/int() usage**    | Casting tensors to Python scalars breaks the graph, forces CPU copies, and rounds twice                  | Same                                                                                  | Banned via EmberCoach/EmberLint Strict                                                                                            |
| **numpy.fft**              | Promotes to float64 and runs on CPU                                                                      | N/A                                                                                   | Any parity test touching NumPy is invalid; removed                                                                                |
| **Device fallbacks**       | MPSGraph FFT still has known accuracy TODO in PyTorch                                                    | MLX has custom Metal FFT                                                              | Even with matching formulas tiny MPS vs Metal diffs remain; solved via deterministic kernels and optional extended-precision path |

## What `float()` / `int()` Do (and Why We Banned Them)

1. `float(tensor)`, `int(tensor)`, `.item()` all synchronise to host, convert to a Python scalar, and round once in host
   FP64 → FP32 when the value is copied back. Graphs lose laziness and Metal buffers are reallocated.
2. Any subsequent tensor op therefore re-ingests a rounded scalar, creating a second rounding even if the formula is the
   same.
3. Fix: create backend scalars on device (`torch.tensor(..., dtype=torch.float32, device=device)` or
   `mx.array(..., dtype=mx.float32)`) and use backend ops (`torch.add/mul/div`, `mx.add/multiply/divide`).

We codified this with **EmberCoach** and **EmberLint Strict** – they error on any Python numeric in compute, except
plain integers in indexing expressions.

## FFT Normalisation Rules

### Torch-like profile (`MLX_M2_PROFILE=torch_like`)

```python
k_f = torch.fft.rfft(k_time, n=2 * L) / (2 * L)
y = torch.fft.irfft(u_f * k_f, n=2 * L, norm='forward')[..., :L]
```

Manual 1/n on the spectrum, no scaling on the inverse.

### MLX-stable profile (`MLX_M2_PROFILE=mlx_stable`)

```python
k_f = mx.fft.rfft(k_time, n=2 * L)
y = mx.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # MLX irfft applies 1/n
```

Inverse handles the single 1/n. Any other combination is wrong. EmberCoach now spots FFT calls and explains which side
of the pair carries the 1/n.

## Where Drift Starts (Tracer Results)

`experiments/lab/trace_monarch_divergence.py` logs MLX vs Torch at every stage:

```
Stage             max_abs    rel_err   max_ulp
----------------------------------------------
in_linear         7.15e-07   1.14e-07  24   ← GEMM order/FMA
...
hyena_out        5.32e-06   4.82e-07  79   ← FFT + bias (after fixes)
```

With strict combine/normalisation the Hyena output differences fall to ~1e-6 relative and ~1e-13 MSE (float32 epsilon).
Any larger spike means we hit a wrong normalisation or a CPU detour.

## Extended Precision Plan

Goal: keep everything in extended precision (double-double) until the very end.

1. **Metal kernels (shared with SciPy-MLX):**
    - `experimental/metal_bitexact/ComplexMul.metal` (strict FP32) → extend to double-double.
    - `experimental/metal_bitexact/Depthwise3.metal`.
    - Upcoming: `fft_pass1d_dd.metal`.
2. **MLX wiring:** HyperProfile flags `ep_freqmul`, `ep_depthwise`, `strict_kernels`.
3. **PyTorch (MPS) parallel:** custom MPS kernels calling the same MSL code so Torch can opt-in.
4. **SciPy-MLX reuse:** expose dd FFT / complex multiply / depthwise operations via C API for the SciPy port.

Benefits: single rounding at the boundary, deterministic accumulation, traceable numerics over billions of ops.

## Guardrails Now Enforced

- No Python scalar arithmetic in compute paths (EmberCoach raises errors).
- No `.item()`, `float()`, `int()` on tensors (error).
- No NumPy in parity paths (promotes to float64 + CPU hop).
- Consistent device selection; bias/kernel all created on the same device.
- Profiles make combine/normalisation explicit and reproducible.

## Action Items

1. Finalise double-double Metal kernels and surface them via HyperProfile (EP modes).
2. Port the same kernels to a Torch MPS extension so upstream users can opt-in to deterministic EP.
3. Keep EmberCoach/EmberLint Strict in CI for both MLX and Torch code.
4. Track long-run drift with the tracer plus periodic CPU reference rebase in DN/EP modes.

These steps bring GPU behaviour within float32 epsilon *and* give us a path to deterministic, extended-precision math
for long-running workloads and the SciPy-for-MLX project.
