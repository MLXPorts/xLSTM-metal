# Research Notes and Findings (xLSTM on Apple/MPS)

Date: 2025-08-28
Owner: xLSTM Apple/MPS track

Purpose

- Capture the working theory, experiments, evidence, and decisions behind our Apple‑centric xLSTM inference stack so
  future contributors can understand the “why” as well as the “what”.

Scope (what this file covers)

- Canon semantics vs. implementation details (fusion, tiling, safety guards).
- External Metal experiments we’re borrowing patterns from.
- Two new research directions we prototyped (documented, quarantined): CfC‑style head and Dendritic Comb Codec (DCC).
- ANE deployment notes and export path.

Core decisions (stable)

- Canon equals: fixed logical `chunk_size`, strict time order, fp32 state for (C,N,M). Runtime shrinking is
  non‑canonical; keep it OFF by default (safety escape hatch only).
- Pseudo‑kernel via `torch.compile`: the per‑timestep “step” is the fusion boundary; the driver handles tiling/banding.
  Goal is to enlarge fusion windows without changing semantics.
- Divide‑and‑conquer for Metal limits: any sub‑chunking (“inner tiling”) is an internal execution detail. Always
  propagate exact state across tiles.
- Lifecycle/telemetry: watchdog + clean Ray shutdown + optional xltop; default to in‑process queued or Ray local_mode=1
  on Apple.

What we validated (evidence)

- Pseudo‑kernel fusion works on MPS: step+sequence compile and run GPU‑only; fusing per‑timestep ops reduces kernel
  count and memory traffic. See: docs/PYTORCH_MPS_FUSION_NOTES.md, docs/PYTORCH_MPS_INFERENCE_ARCHITECTURE.md.
- Ray hygiene: lingering daemons can pin UMA; auto‑shutdown in the driver + `ray stop --force` restores memory when
  needed; dashboard is opt‑in and safe (actors terminated before keep‑alive head).
- Telemetry ring (device→host): cheap to add in compiled step at tile boundaries; gives counters (e.g., NaN/clamps)
  without round‑trips. This mirrors a Metal log buffer pattern.

External Metal inspirations (what we borrow, not literal code)

- MetalCoroutinesTest/NeuromorphicKernel.metal shows:
    - Tiled recurrent multiply (threadgroup buffers) → we let Inductor do this when it can.
    - Exponential gates with a running normalizer n (subtract‑n variant); candidate sigmoid; CfC hidden update:
      `h_new = (h + Δt·(o·sigmoid(c_new)))/(1 + Δt·λ)`.
    - Double buffering of hidden state; device log buffer; optional atomics (for Hebbian) — atomics are NOT adopted for
      inference.
- Mapping: orchestrator discipline (single owner), explicit preallocation, optional device telemetry, optional
  sanitization at tile boundaries.

New experiments (quarantined; not in default path)

1) CfC‑style head (ATen‑only; fusable)

- File: mlstm_kernels/torch/experiments/cfc_head_experiment.py
- What: exp gates (subtract‑n), cell update, optional masks, and CfC hidden smoothing; optional device counter ring.
- Why: to test a continuous‑time smoothing head under `torch.compile` without raw Metal and without altering canonical
  xLSTM by default.
- Status: documented in docs/EXPERIMENT_CFC_HEAD.md; not hooked into core drivers.

2) Dendritic Comb Codec (biological) — tensorized

- File: mlstm_kernels/torch/experiments/dcc_biological_experiment.py
- What: vectorized encode/ decode using η=(0.5)^1.5 and a threshold τ, returning (residue, carries[level]).
- Why: perfect (or near‑fp32) reconstruction with sparse per‑level “events”; candidate for offline weight encoding and
  boundary state encoding.
- Quick run (MPS) — satisfying outputs:
    - Self‑test: True
    - Synthetic 1×4×64 block: max reconstruction error ≈ 2.98e−08; mean carries/element ≈ 0.02897 (sparse); activity at
      level 0 only for τ=0.2 (as expected).
    - Tiny vector [0.0, 0.05, 0.21, −0.35, 0.65] encoded into clear residue+excess; decode matched inputs within fp32.
- Status: documented in docs/EXPERIMENT_DCC_BIO.md; a weight‑slice probe and boundary probe are ready to add next.

Notes (rhythms and packets)

- non_commutative observation: 40‑periodic closure (LCM(5,8)); suggests rhythm‑aware inspection windows (e.g.,
  checkpoint every 40 steps) without affecting fused math.
- core.py: normalization → encode carries → perfect reconstruction → de‑normalize; optional 40‑bit packet to tag events.

ANE deployment (deployment track, not dev loop)

- ane_transformers: HF‑compatible ANE‑optimized modules for on‑device.
- Executorch → Core ML: PyTorch export to a Core ML model for ANE/GPU/CPU; keep dev on MPS, export for production.
- See: docs/ANE_DEPLOYMENT_GUIDE.md.

Open questions (to guide future work)

- Fusion window “inner tiling”: where is the stable sweet spot for T_inner on M3 Ultra (4? 8? 16?) given Inductor+MPS
  limits?
- Parity harness: small shapes, long S — codify exact equivalence across step×1 vs step×T_inner; queued vs ray; clamp
  on/off; sanitizer on/off.
- DCC vs limb precision at boundaries: which offers better reconstruction/overhead trade‑offs for (C,N) on long S? (
  Initial hunch: limb L=2 is near‑fp32 at minimal cost; DCC gives better event semantics and offline compression.)
- Rhythm‑aware checkpoints: does a 40‑step cadence reduce tail risks (numeric drift, telemetry cost) measurably without
  throughput loss?

Next steps (low risk, high value)

- Add a parity test script (small randomized shapes) and wire `--canon-mode` (shrink off, fp32 state checks).
- Add a T_inner compile‑probe (ratchet unroll only at compile time; do not change logical chunk).
- Add a weight‑slice DCC probe and a boundary state A/B (debug flag) to measure reconstruction error and per‑level
  sparsity.

Repro commands (used in this cycle)

- DCC quick test (MPS):
    - PYTHONPATH=. conda run -n base python -c "from mlstm_kernels.torch.experiments.dcc_biological_experiment import
      dcc_self_test;print(dcc_self_test())"
    - PYTHONPATH=. conda run -n base python /tmp/run_dcc_test.py # see saved snippet in notes
- xltop telemetry: conda run -n base python scripts/xltop.py --json
- Ray cleanup (if needed): conda run -n base ray stop --force

Appendix: status tags (for readers)

- Core: compiled MPS step+sequence; chunkwise schedulers; watchdog; Ray auto‑shutdown; xltop.
- Edge: inner tiling (doc’d); device telemetry ring (doc’d); sanitizer (planned flag); compile‑probe (planned).
- Experimental: CfC head; DCC tensorized; ANE export path; limb‑precision state.
