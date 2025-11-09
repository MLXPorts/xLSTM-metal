Bit‑Exact Metal Kernels (Experimental)

This directory contains tiny Metal kernels to isolate numeric differences due to
accumulation order/FMA across frameworks:

- `ComplexMul.metal` — Complex multiply without FMAs and with a fixed evaluation order.
- `Depthwise3.metal` — A deterministic 3‑tap depthwise 1D convolution that mirrors
  the manual depthwise path used in the MLX Monarch mixer tests (padding by 2, window [t, t+1, t+2]).

Usage

- These are intended for micro‑benchmarks or bit‑equality checks. A minimal Swift/Obj‑C
  host can enqueue the kernels and compare results to MLX/PyTorch operations on the same inputs.
- Guard their use behind a “strict” profile toggle when wiring into higher‑level modules.

