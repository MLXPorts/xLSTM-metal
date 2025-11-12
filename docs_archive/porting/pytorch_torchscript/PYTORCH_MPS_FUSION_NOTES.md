# PyTorch/MPS Fusion Notes (Pseudo‑Kernel Design)

This document captures how we implement xLSTM on Apple/MPS using PyTorch’s compiler as a “pseudo‑kernel” system, why it
works, and how we keep behavior canonical while navigating Metal limits.

## Overview

- We run xLSTM on Apple Silicon (MPS) with compiled PyTorch functions (no handwritten Metal).
- The compiled “step” is our main fusion unit; sequence execution is a compiled loop (or a tiled driver) that preserves
  strict time order.
- Fully fused chunkwise kernels can hit Metal per‑kernel argument limits on long sequences; we sidestep this by
  launching many small compiled step kernels across head‑bands and sequence tiles (queued/ray drivers).

## Pseudo‑Kernel via `torch.compile`

- Treat the compiled step function as a kernel boundary: compose all per‑timestep math inside it (gating; (C,N,M)
  update; readout; optional projection).
- Keep dtype/device casts out of the fused region; pre‑cast inputs and maintain consistent strides/layout to avoid graph
  breaks.
- Cache compiled artifacts by shape key: `(B, heads_per_band, Dq, Dv, dtype, device, T_inner)`.

## Inner Tiling (T_inner) to Grow Fusion Windows

- Instead of compiling a scalar “one‑step” kernel, we compile an unrolled block of `T_inner` steps (e.g., 4 or 8) that
  the driver calls repeatedly over a logical sequence tile.
- Benefits: larger fusion regions and fewer MPS kernel launches, while respecting Metal’s argument/graph limits.
- Driver still iterates strictly in time order; (C,N,M) is passed exactly across sub‑tiles and chunk boundaries.

## Canon Semantics (Fixed Logical Chunk)

- Logical `chunk_size` is fixed per run (hyperparameter). Any sub‑chunking or `T_inner` unrolling is an internal
  execution detail.
- State math: keep (C,N,M) updates in float32 for stability; activations/IO can be bf16.
- Ordering: strict time order within each head‑band; bands parallelize safely (no cross‑band state).
- Shrink: runtime chunk shrink is non‑canonical; retained only as an OOM escape hatch on UMA. Recommended default is OFF
  on high‑memory systems.

## Scheduling (What the Drivers Do)

- Heads → bands: split heads into bands; each band maintains its own (C,N,M) and advances in time.
- Sequence → tiles: user‑visible logical `chunk_size`; internal `T_inner` unroll inside the compiled region; loop over
  tiles.
- Queued vs Ray:
    - Queued (in‑process): minimal overhead on Apple; avoids multiprocess duplication.
    - Ray in local_mode=1: ergonomics similar to queued; multiprocess is optional for dashboard/pipelines.

## Metal Limits and Compile‑Probe

- On startup (or first call), probe compile with the requested `T_inner`; if the MPS compiler signals arg/graph limits,
  ratchet `T_inner` down to the next safe value.
- This keeps the logical chunk fixed and only adapts the unroll factor used inside the compiled region.

## Instrumentation

- Fusion introspection: `TORCH_LOGS=+inductor` and `torch._dynamo.explain()` help spot graph breaks and kernel counts.
- Optional `--fuse-report` (planned) to dump a one‑tile fusion summary for debugging.

## Equivalence Tests (Planned)

- Small randomized cases verifying that:
    - step×1 loop ≡ step×T_inner unrolled (allclose on H and terminal (C,N,M))
    - queued ≡ ray drivers under identical shapes
- These prove sub‑chunking/tiling is behavior‑preserving (canonical).

## Defaults for Apple 256 GB UMA

- Keep shrink OFF (safety‑only knob available); set high absolute soft/hard caps if needed.
- Start with `chunk_size ∈ {64, 96}`, `heads_per_band ∈ {2, 4}`, `T_inner ∈ {4, 8}`; pick the fastest via a short sweep.

