# Deploying xLSTM on Apple ANE

This guide summarizes two ANE‑centric paths that complement our PyTorch/MPS development flow.

## 1) ane_transformers (Apple library)
- What it is: a library providing optimized Transformer modules and Hugging Face‑compatible model classes for the Apple Neural Engine (ANE). It includes a reference PyTorch implementation and ANE‑tuned variants.
- Why use it: significantly lower memory use and higher throughput on ANE versus baseline CPU/GPU implementations for supported blocks.
- How it fits our stack:
  - Keep training/experimentation in PyTorch+MPS (our compiled pseudo‑kernels).
  - For on‑device deployment, replace relevant HF modules with ane_transformers equivalents; configure runtime to target ANE.
- High‑level steps (illustrative):
  - Install ane_transformers and its runtime deps.
  - Swap/import the ane_transformers model classes where appropriate.
  - Select ANE compute units in the runtime config; validate parity on small suites.

## 2) Exporting to Core ML via Executorch
- What it is: an export/compilation path that takes a PyTorch model through Executorch and produces a Core ML model that Core ML can schedule on ANE/GPU/CPU.
- Why use it: Core ML specializes and partitions the model for Apple hardware; ANE execution is handled by Core ML’s delegate.
- How it fits our stack:
  - Start from the PyTorch inference graph we already validated on MPS.
  - Use the Executorch export path to prepare an artifact suitable for Core ML.
  - Compile to Core ML and run on device with ANE enabled (subject to operator coverage).
- High‑level steps (illustrative):
  - Export PyTorch graph (e.g., via `torch.export` / Executorch tooling) with fixed shapes/dtypes used in production.
  - Convert/compile to Core ML with ANE compute units selected; run conformance tests and latency/throughput profiles.

## 3) Design considerations for ANE
- Favor fused, memory‑friendly blocks (layer norms, attention variants supported by ANE).
- Keep dtypes/shapes within ANE coverage; avoid exotic control flow at runtime.
- Profile early with realistic sequence lengths; validate numerical parity on a representative corpus.

Notes
- Our primary development loop stays in PyTorch+MPS (GPU) with `torch.compile` for fusion. For production on Apple devices, use ane_transformers and/or Executorch→Core ML to target ANE.

