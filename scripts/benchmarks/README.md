# benchmarks

Benchmark throughput and latency for xLSTM on MPS.

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12
version from conda.**

Tools

- `benchmark.py`: Generic perf harness for quick tests.
- `xlstm_ultimate_benchmark.py`: Comprehensive suite across sizes/devices.
- See also `scripts/bench_mps.py` and `scripts/bench_mps_ray.py` for backend sweeps.

When to use

- Compare chunk_size/head-banding impacts on prefill/decode.
- Verify tuning results from `optimize_mps.py`.
