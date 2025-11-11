# Limb Precision for Recurrent State (bf16×L expansions)

Summary

- We preserve canonical xLSTM math while reducing fp32 state bandwidth by storing the long‑lived recurrent states (C, N)
  as fixed expansions of low‑precision “limbs” (Shewchuk‑style expansions). Default: L=2 limbs in bf16 (double‑bf16 ≈
  fp32‑like), with optional L=3 for extreme contexts. All hot math remains in fp32 inside the fused step; encode/decode
  happens at tile boundaries only.

Why (problem→design)

- Problem: bf16/fp16 accumulators drift over long sequences; fp32 state is stable but heavy in bandwidth and storage.
- Design: compute tiles in fp32 (fused step×T_inner), then compress C,N to bf16×L limbs at tile boundaries. On the next
  tile, decode to fp32, update, and re‑compress. This keeps fusion stable and slashes fp32 I/O.

Representation

- ExpansionL: x ≈ Σ_{i=0..L-1} limb[i], with non‑overlapping mantissas after renormalization.
- Recommended: bf16 limbs (fp32‑range exponent, 7‑8 bits mantissa). L=2 (“double‑bf16”) approximates fp32 well for our
  workloads.

Algorithms (high‑level, ATen‑only)

- Encode (fp32→bf16×L):
    1) limb0 = x.float().to(bf16)
    2) r0 = (x - limb0.float())
    3) limb1 = r0.to(bf16)
    4) (optional) repeat to limb2...
    5) renorm(limbs): bubble carry so |limb[k]| ≪ ulp(limb[k-1])
- Decode (bf16×L→fp32): sum limbs in float32: x = Σ limb[i].float()
- Add/FMA “into expansion” (tile‑internal fp32 math): run fused tile in fp32; only boundary encode/decode uses the
  expansion ops. (Pure limb arithmetic each step is possible but slower.)

Integration points

- At tile start: C_fp32,N_fp32 = decode(C_limbs,N_limbs)
- Run fused step×T_inner in fp32; update (C,N,M); compute H
- At tile end: encode(C_fp32,N_fp32) → (C_limbs,N_limbs)
- M stays in fp32 (or 2‑limb experimental) — single scalar per head

Shapes & layout

- Store limbs in a tail dimension: (B, NH, DHQK, L) and (B, NH, DHQK, DHHV, L) for C if packed; or separate tensors per
  limb. Favor contiguous layout.

Complexity & overhead

- Boundary only: O(L) elementwise work per state tensor per tile, amortized by T_inner. Target <5–8% throughput cost for
  L=2 with T_inner∈{4,8}.
- Memory: fp32 (4B) vs double‑bf16 (2×2B = 4B). L=3 → 6B; trade bandwidth for stability.

Numerical expectations

- L=2 (bf16×2): ≈20–22 effective mantissa bits empirically → near‑fp32 for our use.
- L=3: margin for adversarial long sequences.

Config (planned)

- `XLSTM_STATE_EXPANSION_L={0|2|3}` (0=off default; 2 recommended; 3 expert)
- `XLSTM_STATE_LIMB_DTYPE={bf16|fp16}` (bf16 default)

Validation plan

- Micro: TwoSum/TwoProd style tests over random vectors/matrices; ulp/rel‑error vs fp32 add/FMA/accumulate.
- Macro: full xLSTM forward on small shapes with long S; check allclose on H and terminal (C,N,M) vs fp32.
- Throughput: tok/s with L∈{0,2,3}, T_inner∈{4,8}, chunk_size∈{64,96}.

Notes

- All algorithms written with standard ATen ops so Inductor can fuse around them; we do not rely on exact FMA rounding.
- This is a bandwidth/accuracy trade — keep a “canon mode” (expansion off) for debugging and reference parity.

