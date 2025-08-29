# Neuronal‑Dendrite Comb‑Filter Probe

Origin: Links Rall’s 3/2 branching rule to the Δ‑frequency/Turing‑carry idea.
- Each fork behaves like a low‑Q notch (comb) filter; peaks above a threshold are clipped and stored as carries.
- Rall’s law ⇒ daughter impedance match ⇒ power partition ≈ (1/2)^{3/2} ≈ 0.353.
- We map a π‑derived diameter stream to fork sizes, simulate passive propagation of a δ‑band carrier, and show carries and perfect reconstruction.

Run
- CLI: `python -m lab.dendritic_comb_probe --out runs/comb_probe` (saves waveform/carries/spectrum PNGs)
- Options: `--depth`, `--threshold`, `--noise`, `--no-plots`

Evidence
- Test: `tests/test_dendritic_comb_probe.py` verifies near‑zero RMS reconstruction error.
- Journal: add figures under a future run entry.

Invariants Mapping
- Preserve→Gate: clipping stores carries; residue preserves value; reconstruction proves no loss.
- Time Writes The Story: carries per fork correspond to phase buckets; chain depth of 5 mirrors Z5 carry windows.
- Determinism: π half‑digits seed diameter series; fallback RNG path is also deterministic.

Notes
- Optional optics “tilt‑incidence” demo remains commented; integrate once helpers are present.

