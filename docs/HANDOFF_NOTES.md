# Handoff Notes — Current MPS Compiled Backends (2025-08)

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

This project now runs xLSTM efficiently on Apple Silicon using compiled MPS backends with a unique chunkwise scheduling approach. This note is the authoritative, up‑to‑date entry point for maintainers.

## High‑Level Architecture
- Model: `xlstm_official_full/xlstm_large` (mLSTM blocks), loaded via `from_pretrained` or local sharded safetensors.
- Kernels (inference):
  - Step kernel: `metal` → implemented with `torch.compile` on MPS, not handwritten Metal.
  - Sequence kernel: `native_sequence__metal` (MPS‑compiled step in a loop for decode).
- Chunkwise (prefill) backends on MPS:
    - `chunkwise--queued_compiled_steps`: CPU thread pool queues many small compiled step kernels; all math on GPU.
    - `chunkwise--ray_compiled_steps`: Ray actors (local_mode) coordinate compiled steps; helpful for pipeline/concurrency.
    - `chunkwise--native_compiled_autograd`: Fully compiled chunkwise kernel (PyTorch inductor) as a comparator.

## Entry Points
- Quick run (local HF checkpoint): `scripts/run_local_xlstm_mps.py`
  - Select chunkwise backend: `--chunkwise-backend {queued_compiled_steps,ray_compiled_steps,native_compiled_autograd}`
  - Tunables: `--chunk-size`, `--workers` (queue threads), `--heads-per-band`, `--streams` (optional MPS streams)
- Simple check: `run_xlstm.py` (uses step=metal, sequence=native_sequence__metal)
- Tuning/bench: `scripts/optimize_mps.py`, `scripts/bench_mps.py`, `scripts/bench_mps_ray.py`

## Unique Chunking Approach (Prefill)
- Heads are split into bands; the sequence is split into small chunks (default 32).
- A CPU thread pool (or Ray actors) enqueues many small compiled step kernels to MPS.
- Per‑band sequential order is preserved; concurrency is achieved across bands and small chunks.
- Optional `torch.mps.Stream` contexts encourage overlap; all math stays on GPU.

Scheduler diagram (bands × chunks)

```
Heads (H) → bands of size hpb                 Sequence (L) → tiles of size C

Band 0: h0..h{hpb-1}                          Tiles:  t0   t1   t2   t3   ... t{L/C-1}
Band 1: hh..h{2*hpb-1}
...

Dispatch matrix (Ray actors A_k per band):
            t0     t1     t2     t3   ...
  band 0:  A0 →   A0 →   A0 →   A0 →  ...   (sequential within band)
  band 1:  A1 →   A1 →   A1 →   A1 →  ...
  ...

Parallelism across bands at each tile; per‑band order is preserved across tiles.
Driver stitches h_out and last states (C, N, M) per band.
```

Key implementation:
- Compiled step: `mlstm_kernels/torch/recurrent/metal/compiled.py` (`mlstm_recurrent_step__metal`) — enforces float32 math and stabilized gating.
- Queued chunkwise: `mlstm_kernels/torch/chunkwise/queued_compiled/driver.py` — bands × chunks scheduling with thread pool and optional streams.
- Ray chunkwise: `mlstm_kernels/torch/chunkwise/ray_compiled/driver.py` — same idea with Ray actors.
- Fallbacks are not used in these paths; MPS is required for the `metal` backend.

Ray specifics (in code):
- Default `XLSTM_RAY_LOCAL_MODE=1` keeps execution in‑process (no tensor copies), GPU‑only.
- Actor: `HeadBandWorker` processes one head band over a sequence slice; returns band outputs and last states.
- Scheduler: driver tiles sequence into `chunk_size` segments and dispatches per‑band actors across all tiles; aggregates with `ray.wait`.
- Tunable: `XLSTM_MPS_HEADS_PER_BAND` controls parallelism granularity; default 4.

Queued specifics (in code):
- CPU thread pool of size `XLSTM_MPS_WORKERS` (default 6) launches per‑band loops, stepping through chunks.
- Optional `torch.mps.Stream` contexts (`XLSTM_MPS_STREAMS`) to encourage overlap; guarded by build capability.
- Micro‑autoscale (`XLSTM_MPS_AUTOSCALE=1`) probes and may halve `heads_per_band` if a tiny probe exceeds 10ms.

## Configuration and Env Vars
- Script args map to env vars consumed by backends:
  - `XLSTM_CHUNKWISE_BACKEND` → selects chunkwise backend key.
  - `XLSTM_MPS_WORKERS` → CPU coordinator threads (queued backend).
  - `XLSTM_MPS_HEADS_PER_BAND` → heads per band (parallelism granularity).
  - `XLSTM_MPS_STREAMS` → number of MPS streams (optional; many builds lack stream support).
  - `XLSTM_MPS_AUTOSCALE` → enable micro autoscale of heads_per_band (queued backend; default off).
  - `XLSTM_RAY_LOCAL_MODE` → run Ray in local_mode=1 (in‑process; default on).
  - Model config defaults for MPS (set in `scripts/run_local_xlstm_mps.py:load_local_config`):
  - `mode="inference"`, `return_last_states=True`, `autocast_kernel_dtype="bfloat16"`, `inference_state_dtype="float32"`.

Step math details (from compiled step):
- Uses stable log‑sigmoid for forget gate, log‑sum‑exp style stabilization for exponentials.
- Forces float32 for all gate/state math; q scaled by `DHQK**-0.5`.
- Denominator uses `max(|q·N|, exp(-m)) + eps` to avoid under/overflow.

## Recommended Defaults (M3‑class GPUs)
- Ray backend (default): `--chunkwise-backend ray_compiled_steps --chunk-size 32 --heads-per-band 4` and `XLSTM_RAY_LOCAL_MODE=1`
- Queued backend (legacy): for environments without Ray, `--chunkwise-backend queued_compiled_steps --chunk-size 32 --heads-per-band 4 --workers 6`

## Key Files
- `xlstm_official_full/xlstm_large/model.py` — model + config fields (`chunkwise_kernel`, `sequence_kernel`, `step_kernel`).
- `mlstm_kernels/torch/chunkwise/__init__.py` — backend registry keys wired to implementations.
- `mlstm_kernels/torch/recurrent/metal/compiled.py` — MPS step kernel (compiled PyTorch), numerically stabilized.
- `mlstm_kernels/torch/chunkwise/queued_compiled/driver.py` — queued chunkwise driver.
- `mlstm_kernels/torch/chunkwise/ray_compiled/driver.py` — Ray chunkwise driver.

## Notes
- “metal” naming indicates MPS‑compiled execution; there are no handwritten Metal shaders in the inference path.
- The previous docs referring to incomplete custom Metal shaders are obsolete; this file reflects the current working design.

## Quick Start
```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTHONPATH=. \
python scripts/run_local_xlstm_mps.py \
  --model_path ./xlstm_7b_model \
  --chunkwise-backend queued_compiled_steps \
  --chunk-size 32 --heads-per-band 4 --workers 6 \
  --prompt "The capital of France is" --max_new_tokens 20
```

## Optimization Harness
- Script: `scripts/optimize_mps.py` — random or simple GA to maximize decode tok/s; logs to `runs/mps_opt/<backend_timestamp_tag>/`:
  - `run_meta.json` (config), `trials.jsonl` (all trials), `summary.csv` (table), `best.json` (best params+metrics)
- Save and rate outputs for best/interesting trials:
  - `scripts/save_outputs_for_trials.py --run <run_dir> --model_path ... --prompt-file ... --new 32 --outputs <run_dir>/outputs`
  - `scripts/judge_outputs.py --model_path ... --prompt-file ... --outputs <run_dir>/outputs`
- Metrics: prefill/decode tok/s, and quality metrics from teacher‑forced scoring (avg_logprob, perplexity) and diversity (distinct-2/3).

## Future Direction
- Favor `ray_compiled_steps` as the default chunkwise backend on Apple MPS.
- Keep `queued_compiled_steps` available for environments without Ray or for simple single‑process benchmarking.

For tuning and sweeps, see `scripts/optimize_mps.py` and `docs/TUNING_GUIDE.md`.
