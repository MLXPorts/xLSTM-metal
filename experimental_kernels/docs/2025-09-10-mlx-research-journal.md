Title: MLX xLSTM Research Journal — 2025‑09‑10

Goal
- Build a high‑performance, numerically stable MLX (Metal) xLSTM path by exploring kernel‑level and model‑level changes using lab‑first protocols. All prototypes land in `lab/` with documentation before any integration. We preserve provenance and explain limitations so future readers understand what works, when, and why.

Scope Isolation
- Experimental code lives under this lab folder.
- The production model remains in `implementations/` and is only modified behind safe defaults or after lab validation.
- This entry captures the current status verbatim and catalogs files created/changed as part of the effort.

Status (verbatim)

Awesome — I kept everything research-first in lab and built/validated new pieces before touching integration.

What I added (lab-first)
- Docs included in this folder (plan/report/journal).
- Atomic lab suite (all standalone) in this folder.
- New lab kernel: Multi-Head LayerNorm (Metal) in `kernels/` and bench here.

Key lab results
- Soft-cap parity: max|Δ|=0.0; performance comparable.
- HeadLinear vs block-diag: exact match; avoids block-diag build cost.
- Head-aware LN: per-head mean≈0/var≈1; different from GroupNorm (by design).
- GEMM tiles (M3): 16x8 and 8x16 faster than 16x16 in our sweep.
- Sequence precompute: ~20% win on demo by reducing per-step dispatch.
- MHLN kernel: parity ~1e‑6; wins for DH≈96–128 in this setup.

Integration status (minimal)
- MLX model updated (HeadLinear, MultiHeadLayerNorm, soft caps; fast-head default OFF). Smoke+parity OK.
- Future kernel swaps will remain behind flags until vetted.

Next
- Add affine/vectorization to MHLN kernel; bench thresholds.
- Prototype vec4 GEMM in lab; compare vs tiled GEMM.
- Wire sequence-precompute flag in model; add bench harness.
- Explore chunkwise prefill prototype.

Provenance
- Files in this folder created from 2025‑09‑10 experiments, with results captured above.

