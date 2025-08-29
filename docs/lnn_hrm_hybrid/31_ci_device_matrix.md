# CI & Device Matrix — Notes

Targets: CPU minimal and MPS (Apple). CUDA is not targeted in this repo configuration. No native fallback for compiled graphs.

## Matrix
- cpu: run CPU-safe unit tests (MemoryCube, trace, scheduler)
- mps: run all tests; mark heavy/integration as `slow`
- cuda: not supported here

## Tips
- Gate device usage via `torch.backends.mps.is_available()` / `torch.cuda.is_available()`
- Export `PYTHONPATH=.` for repo-local imports
