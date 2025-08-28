# xLSTM Inference on Apple/MPS — Architecture and Technique

This document dissects our inference stack on Apple Silicon using PyTorch’s MPS backend. It explains how we build “kernel‑like” graphs with `torch.compile`, why we tile in time/head space, and how we preserve canonical xLSTM semantics while navigating Metal limits. It also contrasts our PyTorch path with the MLX path and summarizes the current state of ANE (Apple Neural Engine) for PyTorch.

## 1) Goals and Constraints
- Canonical math: identical (C, N, M) recurrence as in the xLSTM paper; strict time order; fp32 state math.
- Apple‑friendly: GPU‑only MPS execution; no handwritten Metal; avoid UMA thrash; clean lifecycle.
- Fusion‑first: maximize `torch.compile` fusion (reduce traffic/launches) without tripping Metal’s per‑kernel argument/graph limits.

## 2) Pseudo‑Kernel via `torch.compile`
- We treat the per‑timestep mLSTM cell as the fusion boundary (“pseudo‑kernel”). The compiled step contains:
  - Gate stabilization: `M_new = max(log(sigmoid(F)) + M_old, I)`; scale gates by `exp(… − M_new)`.
  - State update: `C_new = F_act * C_old + I_act * (V ⊗ Kᵀ)`; `N_new = F_act * N_old + I_act * K`.
  - Read/normalize: `H = (C_new Q) / max(Qᵀ N_new, eps)`; optional projection.
- Implementation notes
  - Keep dtype/device casts out of the fused region; pre‑cast to bf16 for activations and ensure (C,N,M) accumulators are fp32.
  - Avoid view patterns that cause graph breaks; use contiguous/layout‑stable tensors.
  - Cache compiled artifacts by a small “shape key”: `(B, heads_per_band, DHQK, DHHV, dtype, device, T_inner)`.

## 3) Inner Tiling (T_inner) — Growing Fusion Windows
- Instead of compiling a scalar one‑step kernel, we `torch.compile` a function that unrolls `T_inner` timesteps (e.g., 4 or 8) in a tight loop over the same band slice. The driver calls this compiled block repeatedly to cover the sequence tile.
- Payoff: larger fusion regions, fewer calls, and lower per‑tile overhead; semantics stay identical because time order is preserved and (C,N,M) is threaded through each sub‑tile.
- Safety: on first compile, we can probe `T_inner` and ratchet down if the MPS compiler signals argument/graph limits. Logical `chunk_size` remains fixed; only the internal unroll adapts.

## 4) Chunkwise Scheduling (Bands × Tiles)
- Heads → bands: split NH into bands of size `heads_per_band` (parallel‑safe; bands do not share state).
- Sequence → tiles: user‑visible logical `chunk_size` (canon); inside the kernel we unroll `T_inner` and loop to cover each tile.
- Backends
  - Queued (in‑process threads): default on Apple; minimal overhead; optional `torch.mps.Stream` when supported.
  - Ray actors (local_mode=1): ergonomic async pipeline; dashboard optional; we auto‑shutdown/terminate actors to avoid UMA leaks.

## 5) Memory & Lifecycle Guards
- Unified‑memory watchdog: sample RSS + MPS allocation; warn at soft; hard‑abort cleanly; shrinking is a non‑canonical escape only.
- UMA etiquette: call `torch.mps.empty_cache()` sparingly (soft‑limit action) and prefer fixed logical tiles; avoid multiprocess buffer duplication unless the dashboard is required.
- Clean shutdown: call `ray.shutdown()` if we started Ray; terminate actors if the dashboard remains alive.

## 6) MLX Path (contrast)
- MLX supports Metal natively with first‑class kernels; many ops compile directly to Metal without extra work. This makes a “pseudo‑kernel” approach less critical.
- Trade‑off: PyTorch gives us Inductor fusion and ecosystem glue; MLX gives simpler, more predictable Metal execution. We keep MLX as a second implementation for portability and validation.

## 7) ANE (Apple Neural Engine) — Library & Export Paths
- ane_transformers (Apple): a library with optimized Transformer blocks and HF‑compatible model classes to deploy on ANE. It includes a reference PyTorch implementation and ANE‑tuned variants that reduce memory and improve throughput versus baseline CPU/GPU paths.
  - Where it fits: for on‑device deployment, not development‑time fusion; pair with Core ML/ANE execution.
  - Integration pattern (high‑level): swap HF modules for ane_transformers classes and select ANE compute units in the runtime configuration.
- Export to Core ML with Executorch: PyTorch models can be exported through Executorch into Core ML for ANE execution. Core ML’s compilation step optimizes and partitions the graph for Apple hardware (ANE/GPU/CPU).
  - Where it fits: production deployment on Apple Silicon devices; not used during our torch.compile MPS development loop.
- Design guidance: certain ops/architectures benefit more from ANE; consult ANE/Core ML operator coverage and design to maximize ANE utilization (e.g., fused attention, memory‑friendly norms). For development, we continue to target MPS GPU; for productization, use ane_transformers and/or Executorch→Core ML.

## 8) How to Build “Kernel‑like” Graphs in PyTorch (Practical Tips)
- Keep Python control flow out of the compiled region; perform it in the driver.
- Pre‑cast inputs; keep state dtype fixed (fp32); avoid implicit upcasts/downs.
- Prefer contiguous tensors and stable strides across calls; pre‑allocate outputs to avoid allocator noise.
- Bound specializations: fix `B`, `heads_per_band`, and `T_inner` to keep the compile cache small.
- Instrument fusion: `TORCH_LOGS=+inductor`, `torch._dynamo.explain(fn)`; add per‑tile timers; watch kernel counts.

## 9) Canon Mode (Recommended Defaults)
- `chunk_size` fixed; `T_inner` compiled; shrink OFF; fp32 for (C,N,M); strict per‑band time order.
- High‑memory Apple systems (256 GB UMA): start with `chunk_size ∈ {64, 96}`, `heads_per_band ∈ {2, 4}`, `T_inner ∈ {4, 8}`; pick fastest.

## 10) Further Reading
- Apple: Metal/MPS backend notes for PyTorch; Apple ML Research on deploying Transformers on ANE via Core ML.
- PyTorch issue threads re: MPS/ANE routing and performance on macOS.

Notes
- This doc summarizes our technique and the state of tooling as of 2025‑08‑28.
