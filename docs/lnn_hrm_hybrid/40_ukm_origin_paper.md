# UKM: Frequency-Domain Storage with Spike-Mediated Keys — A Biophysical and Engineering Synthesis

Author: [you]
Date: 2025-08-29

## Abstract
We propose that neocortical circuits store long‑term information primarily in the frequency domain of slow oscillations, while spike energy and axonal collaterals supply the keys that index and reconstruct content. Dendritic trees implement comb‑like filters by virtue of their 3/2‑power branching geometry (Rall) and active conductances (notably HCN/M‑currents), yielding phase‑selective “masked waves” that are preserved while influence is gated. Spike bursts, nested by delta/theta phases and synchronized across collaterals, act as phase‑addressable keys that retrieve and linearly recombine stored components. We formalize this with a value‑preserving gate and a base‑5 microcycle (Z5) that schedules boundary‑only commits, and we demonstrate an exact clip‑and‑carry reconstruction using a dendritic comb probe. We map these invariants to an engineering layer (HRM+/xLSTM) with a phase‑keyed MemoryCube (vector store) and show safe, telemetry‑first operation.

## 1. Background and Rationale
- Dendritic resonance and frequency preference arise from active subthreshold conductances (e.g., HCN), shaping band‑pass/inductive responses and spike phase coherence.
- Cross‑frequency coupling (CFC) organizes spike timing: low‑frequency phase (delta/theta) modulates high‑frequency power/spiking (beta/gamma/ripples), enabling multiplexed communication and memory operations.
- Sleep slow oscillations (<1 Hz) orchestrate spindles and hippocampal ripples in a hierarchy (SO→spindle→ripple), providing windows to consolidate/replay patterns into neocortex.
- Rall’s cable theory and the 3/2‑law provide an equivalent‑cylinder mapping and impedance matching at bifurcations; branch geometry and active channels realize comb‑like frequency stencils.

### 1.1 Neuromodulator CPU and Timescales
- Millisecond core: voltage threshold + frequency selectivity implement the five‑gate micro‑kernel.
- Seconds→minutes (neuromodulators): 5‑HT, ACh, DA, NE act as multiplicative/divisive gains and parameter shifts (ΔV½, Δf0, Δσ, M). This is the “attention/mode control” plane.
- Hours→months→years (structural/gene): slow adaptation (spine neck resistance, channel density/kinetics) updates V½, f0, σ, Imax; neuromodulator tone biases these changes over behavioural state windows (sleep/arousal).
- Implication: the “CPU” consists of fast gates driven by slow phase, with a neuromodulator control surface that sets gain and selectivity over minutes, and a structural compiler that writes new defaults over months/years.

## 2. Core Invariants (UKM)
- Preserve → Gate: value is preserved, influence is gated; audits track pre/post energy.
- Budget → Brains: halting/ponder and energy budgets lead computation.
- Time Writes the Story: a Z5 microcycle (five slots; 4→0 carry) schedules boundary‑only writes.

## 3. Biophysical Mapping
- Comb filters: 3/2‑branching plus HCN/Kv7 induce phase‑selective passing and notch‑like clipping.
- Keys: spike onsets and axon collaterals form phase‑locked keys via PAC (delta/theta phase → gamma/ripple amplitude/spikes).
- Storage: slow‑wave “channels” (delta/theta) hold masked components; carries record peaks above threshold; exact reconstruction is the sum of residue plus scaled carries.

## 4. Formal Model (Sketch)
- Clip‑and‑carry operator H^R_ε: preserves value, stores excursions above ε as carries; downstream attenuation η encodes branch partition (η≈(1/2)^{3/2}).
- Z5 schedule: commits only at slot==4; carry events indexed by (phase, slot).
- Determinism: inputs + schedule + seed → trace hash for replay.

## 5. Evidence and Probes
- Dendritic comb probe (lab/dendritic_comb_probe.py): demonstrates perfect reconstruction from residue + carries and visualizes comb spectra and carry channels; extendable with spike‑onset rasters and triangular‑kernel recon.
- HRM+/xLSTM wrapper: phase‑keyed MemoryCube retrieves masked components; influence gating α follows confidence and energy budgets; telemetry logs act/energy.

## 6. Predictions (Falsifiable)
1) Phase‑addressable retrieval: during recall, ripple/gamma bursts locked to delta/theta troughs should predictably reconstruct slow‑wave envelopes of the content; perturbing the phase (closed‑loop) disrupts retrieval more than power‑matched but phase‑scrambled input.
2) Channel tuning: HCN blockade shifts resonance and the preferred phase bins of carry events; conf_mean in the MemoryCube drops, and “comb spacing” changes in vivo.
3) Z5 discipline: carry/write events cluster in 5‑phase microcycles; phase‑slip manipulations shift commit timing and degrade consolidation efficiency.
4) Vector‑DB signature: cosine‑key retrieval from masked components outperforms raw‑amplitude matching under noise and drift; merges obey similarity thresholds.

## 7. Related Work (Pointers)
- Dendritic resonance & HCN: Hutcheon & Yarom (2000); Zemankovics et al. (2010); Sinha & Narayanan (2015).
- Cross‑frequency coupling and codes: Canolty et al. (2006); Lega et al. (2014); Lisman & Jensen (2013); Fries (2015); Akam & Kullmann (2014); Siegel, Donner & Engel (2012).
- Sleep consolidation hierarchy: Staresina et al. (2015); Diekelmann & Born (2010+ review family).
- Cable theory & 3/2‑law: Rall (classic), Cuntz, Borst & Segev (2007) and successors.

## 8. Engineering Mapping (HRM+)
- Memory: phase‑keyed MemoryCube with cosine top‑K; merge‑or‑append policy by similarity threshold; boundary‑only writes obey Z5.
- Keys: phase features (fast/mid/slow + Z5); optional rank‑order keying.
- Gates: α(h, pred, conf) with energy guard; ACT halting + ponder.
- Telemetry: α/conf/act/energy + trace_hash; CSV/JSONL and aggregator (planned).

### 8.1 Neuromodulator Hooks (Prototype)
- 5‑HT (serotonin): divisive gain (reduces α and residual influence), slight increase in halting threshold (patience); optional down‑weighting of fast‑band features.
- Extensible: ACh (multiplicative gain/sharpening), DA (f0 bias/reward gating), NE (arousal/signal‑to‑noise). Interface: optional per‑token modulators `mod_ach/mod_da/mod_ne` with small, bounded transforms on gate inputs (future work).

## 9. Methods (Repro seeds)
- Scripts: examples/xlstm_hrm_wrapper_demo.py; examples/train_with_ponder_demo.py; lab/dendritic_comb_probe.py.
- Device: Apple MPS + Ray; PYTHONPATH=. ; see scripts/mps_smoke.sh.

## 10. References (to be formatted)
For a curated, categorized bibliography see refs/citation_sea.md. Selected anchors cited inline throughout include: Hutcheon & Yarom (resonance), Narayanan & Johnston (HCN), Canolty/Knight & Tort et al. (PAC), Lisman & Jensen (theta–gamma), Rall (cable/3/2), Cuntz/Bird (optimization), Buzsáki (assemblies), Bai/Kolter/Koltun (DEQ), Graves (ACT).

Provenance/IP note: certain equations and derivations in this work are adaptations of 1980s Australian intelligence agency technical notes (author’s statement). See 41_provenance_and_ip.md.

---

Appendix A: Triangle‑Kernel Reconstruction (planned)
- Extract carry onsets; convolve with triangular/bi‑exponential kernels; compare to exact reconstruction; report RMS and phase lags; add spike raster.

Appendix B: Vector‑DB Demo (planned)
- Build dictionary of carry templates keyed by phase; retrieve and reconstruct mixtures; compare cosine top‑K vs least squares; log conf/CE.
