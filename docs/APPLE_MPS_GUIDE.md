Apple/MPS Guide: Running xLSTM on Apple Silicon (GPU-only)

Overview
- This repo provides pure-PyTorch compiled backends for mLSTM and sLSTM on Apple Silicon (MPS), without Triton.
- All math runs on GPU. Compiled kernels are strict: if compilation fails, they raise (no CPU fallback).

Backends (Apple Defaults)
- mLSTM step: `step_kernel="metal"` (compiled step)
- mLSTM sequence: `sequence_kernel="native_sequence__metal"` (compiled loop over step)
- mLSTM chunkwise (prefill): `chunkwise_kernel="chunkwise--queued_compiled_steps"` (GPU-only queued compiled-step driver)
- sLSTM: backend switches to `native_compiled` automatically on MPS (strict compile)

Quick Start (Local Checkpoint)
- Command:
  - `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 \
     python scripts/run_local_xlstm_mps.py \
       --model_path /path/to/xlstm_7b_model \
       --prompt "The capital of France is" \
       --max_new_tokens 64 \
       --workers 6 --heads-per-band 4`

Tuning Knobs (Environment)
- `XLSTM_MPS_WORKERS` (default 6): number of CPU coordinator threads.
- `XLSTM_MPS_HEADS_PER_BAND` (default 4): how many heads each worker processes per task.
- `PYTORCH_ENABLE_MPS_FALLBACK=0`: enforce GPU-only execution.
- Optional logs: `TORCH_LOGS=+dynamo` and `TORCHDYNAMO_VERBOSE=1` for compile debugging.

Notes on Chunkwise Prefill
- Fully compiled chunkwise can hit Metal’s per-kernel argument limits on long sequences in the current prototype MPS compiler.
- The `queued_compiled_steps` backend sidesteps this by dispatching many small compiled step kernels across head bands and sequence tiles, keeping the GPU saturated without hitting kernel-arg limits.

Hugging Face Path (Downloads Model)
- `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_hf_xlstm_metal.py`
- Uses the Apple defaults for step/sequence/chunkwise.

Validation
- mLSTM parity: `PYTHONPATH=. python tools/test_metal_parity.py`
- sLSTM parity: `PYTHONPATH=. python tools/test_slstm_parity.py`

