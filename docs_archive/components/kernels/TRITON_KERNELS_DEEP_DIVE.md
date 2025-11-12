# xLSTM Triton Kernels — Deep Dive (Installed Package Overview)

This document surveys the Triton kernels that ship with the `mlstm_kernels` PyPI package we have installed locally and
explains how they relate to our Apple/MPS path.

- Installed package: `<local_install>/site-packages/mlstm_kernels` (version 1.0.3 detected).
- Scope: CUDA/Triton kernels (NVIDIA) and JAX wrappers; these do not run on Apple/Metal. On Apple we use compiled
  PyTorch MPS backends (no Triton).

## Why Triton kernels exist (context)

- The authors published “Tiled Flash Linear Attention” (TFLA), a tiled kernel suite for mLSTM/xLSTM designed to maximize
  SRAM residency and reduce memory traffic. Public claims compare favorably to FlashAttention‑3 and Mamba on NVIDIA
  GPUs.
- The PyPI package `mlstm_kernels` bundles those optimized kernels and wrapper code for JAX/PyTorch. On CUDA, these are
  required to reach the “faster xLSTM” path; on MPS they are not used.

References:

- TFLA arXiv: tiled kernels for mLSTM/xLSTM (flash‑linear attention style). Faster than FA‑3 and Mamba in the authors’
  graphs (NVIDIA context).
- `mlstm_kernels` on PyPI: optimized kernels “required” for faster xLSTM on supported platforms.

## Kernel Families (by directory)

### 1) Triton recurrent step (fused forward)

- File: `mlstm_kernels/triton/recurrent/fw_step_fused.py`
- Kernel: `recurrent_step_fw_kernel` (Triton JIT)
- Purpose: single‑timestep mLSTM forward in a single fused kernel; updates (C, N, M) and computes H.
- Highlights:
    - Stabilizer M: computes `M_new = max(log(sigmoid(F)) + M_old, I)` and rescales gates with `exp(…−M_new)`.
    - Block tiling across DHQK × DHHV with `tl.make_block_ptr`; loops over DHQK blocks to accumulate H and the q•n dot.
    - Claims a ~2× step speedup vs plain `torch.compile` and ~30% vs non‑fused Triton step in the file header.

### 2) JAX wrappers for Triton step/sequence

- Dir: `mlstm_kernels/jax/recurrent/`
    - `triton_step.py`: exposes `mlstm_recurrent_step__triton_fw` using `jax_triton.triton_call` and sets DTYPE/shape
      tiling.
    - `native_sequence.py` / `native_sequence_scan.py`: `mlstm_recurrent_sequence__triton_step_fused_fw` scans the fused
      step across a full sequence (forward‑only variants, custom grad hooks in `fwbw.py`).
- Purpose: integrate the Triton step into JAX pipelines with custom gradients.

### 3) Chunkwise kernels — “triton_xl_chunk”

- Dir: `mlstm_kernels/jax/chunkwise/triton_xl_chunk/`
    - `fwbw.py`: orchestrates chunked forward/backward; returns H and (optionally) all state tensors for backward.
    - `bw.py` (+ `bw_parallel_dQ.py`, `bw_parallel_dK.py`, `bw_parallel_dV.py`): backprop over chunks with per‑dimension
      parallelization.
- Key parameterization:
    - `chunk_size_inter` / `chunk_size_intra`: control tiling across sequence and feature dims; asserts
      `S % chunk_size_inter == 0`.
    - `autocast_kernel_dtype`, `dtype_state` allow fast math with stabilized float32 state.
- Purpose: long‑sequence efficiency through tiling while preserving the mLSTM recurrence.

### 4) Other kernel families in the package

- There are “limit_chunk” variants (not listed exhaustively here) that favor stricter tiling/limits for large shapes.
- Utilities translate framework dtypes to Triton and set launch sizes (e.g., `jax/utils.py`).

## What we use on Apple/MPS

- We do not run Triton on Apple/Metal; Triton targets CUDA.
- Our path: compiled PyTorch MPS “pseudo‑kernels” (torch.compile) for the mLSTM step and a compiled sequence/driver that
  tiles in time and across head‑bands, preserving exact (C,N,M) semantics.
- We adopt the same execution ideas (tiling/fusion), but through PyTorch Inductor + MPS rather than Triton.

## Mapping between Triton and our MPS path

- “Fused step (fw)”: Triton’s `recurrent_step_fw_kernel` ↔ our compiled step kernel.
- “Chunkwise fwbw”: Triton `triton_xl_chunk/fwbw.py` ↔ our chunkwise drivers (queued / ray) with strict time order and
  per‑band state.
- Stabilizer and dtype policy: both keep (C,N,M) in float32; activations may be bf16.
- Tiling knobs: Triton has explicit chunk inter/intra; we express logical `chunk_size` plus an internal unroll `T_inner`
  in compiled MPS.

## About performance claims (FA‑3 / Mamba)

- The “faster than FA‑3 & Mamba” claims are specific to the authors’ NVIDIA measurements with Triton/CUDA kernels. They
  do not directly apply to Apple/MPS.
- On Apple Silicon, our torch.compile MPS path is the right way to approximate the same ideas; absolute numbers will
  differ by hardware and compiler.

## How to tell if Triton kernels are active

- On CUDA/JAX: imports resolve to `mlstm_kernels/jax/.../triton_*` modules; logs mention `triton_call` and Triton launch
  sizes.
- On our Apple run: no Triton imports or CUDA devices; compiled MPS path is used. You can set verbose Inductor logs (
  `TORCH_LOGS=+inductor`) to inspect fusions.

## Pointers

- Authors’ kernel library (GitHub): source for TFLA kernels and wrappers (look for `triton_xl_chunk` and
  `recurrent/fw_step_fused.py`).
- PyPI: `mlstm_kernels` package with release notes and installation guidance.
- Paper (TFLA): tiled kernels for mLSTM/xLSTM with reported speedups versus FA‑3 and Mamba on NVIDIA GPUs.

*This doc explains what’s installed and how it compares to our Apple/MPS approach. For our design notes on fusion/tiling
in MPS, see `PYTORCH_MPS_FUSION_NOTES.md`.*

