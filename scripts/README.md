# scripts

Utilities and entrypoints for running, benchmarking, optimizing, and debugging xLSTM on Apple Silicon (MPS).

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12
version from conda.**

Subfolders

- benchmarks: Throughput and performance runners (prefill/decode timing).
- build: Build helpers (e.g., extension or toolchain setup scripts).
- checks: Lightweight validators and sanity checks.
- debug: One-off probes and debug tools (MPS memory, dims, storage API).
- downloads: Model/weight download and loading helpers.
- runners: User-facing entrypoints to run/train/infer locally.
- experiments: Isolated sandboxes for algorithmic or scheduling experiments.

Calling graph (high-level)

- runners/run_local_xlstm_mps.py → xlstm_official_full.xlstm_large (model) → mlstm_kernels (compiled MPS backends)
- benchmarks/* → runners/load helpers → model.forward/generate
- optimize: scripts/optimize_mps.py (tuning) → model.forward/generate; save/judge complete the loop.

Diagrams

1) Bands × chunks scheduler (Ray default)

```
heads → bands (size = heads_per_band)
sequence → tiles (size = chunk_size)

actors A_k per band; iterate tiles sequentially per band, parallel across bands

time →
band 0:  A0(t0) → A0(t1) → A0(t2) → ...
band 1:  A1(t0) → A1(t1) → A1(t2) → ...
...
stitch outputs + states across bands per tile
```

2) Entrypoint → Backends

```
run_local_xlstm_mps.py
  └─ config: step=metal (compiled MPS), sequence=native_sequence__metal,
             chunkwise=ray_compiled_steps (default)
      └─ model.forward/generate → mlstm_kernels (compiled step)
```
