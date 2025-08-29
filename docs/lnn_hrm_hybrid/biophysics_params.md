# Biophysics Parameters (Defaults)

These defaults are used in bio-mode demos and can be tuned.

- HCN1 (hyperpolarization-activated)
  - V½ ≈ −85 mV
  - f0 ≈ 6 Hz (theta resonance)
  - σ ≈ 4 Hz (bandwidth)
  - I_max arbitrary unit (scaled in demos)

- Phase Windows (theta cycle ≈ 167 ms)
  - 5 sub-windows W0..W4 ≈ 33.4 ms each
  - Suggested mapping: W0 HCN1/XOR_STORE, W1 Cav3.2/INTEGRATE, W2 Kv7/FILTER_NOT, W3 BK/COINCIDENCE, W4 Nav/OUTPUT_XOR

- Energy Budget
  - Initial E0 per segment ≈ 1000 units
  - Drain ΔE(φ) ∈ {0,1,2}; φ from mod-3 phase feature
  - Soft/Hard thresholds: E_min_soft≈200, E_min_hard≈50

- Spike Distances (comb)
  - Victor–Purpura q ≈ 1.0
  - van Rossum τ ≈ 10 ms
  - Fusion weights w_s=w_d=0.5 (learnable)

- Codebooks / Quantized Ladders
  - 16–17 bins aligned with blocky activation levels
  - SHIFT granularity 2–4 positions; XOR over bin codes

