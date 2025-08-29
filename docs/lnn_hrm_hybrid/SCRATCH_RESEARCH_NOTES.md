# SCRATCH RESEARCH NOTES — HRM × LNN × Memory Cubes × Spike Comb × Plasticity

Status: live scratchpad (will evolve fast). Owner: Sydney + collaborators

This document gathers the active ideas, working formalism, code stubs, open questions, and experiment plans for a hybrid system that blends:
- Hierarchical Reasoning Model (HRM; slow/fast loops)
- Liquid neural dynamics (LTC/CfC-style) as the fast module
- Per-block “memory cubes” as associative neuroplastic caches
- A dendritic “spike comb” encoder/decoder with near-perfect reconstruction from spikes
- Confidence/hysteresis gating for safe mentee→mentor takeover
- Optional neuroplastic fast-weights (Oja/BCM/STDP-flavored) with decay and audits

This is the single sheet we will keep editing as the “pills” land.

---

## 0. Rules of Engagement (Scratch)

- Keep this document self-contained: key formulas, pseudocode, diagrams, and API shape.
- Mark unknowns as TODO[spec] and TODO[impl].
- Assume canonical mode unless stated: fixed logical chunk_size, strict time order, fp32 states.
- Safety first: bounded updates, audits, UMA-aware memory policy.

---

## 1. Architecture Snapshot

```mermaid
flowchart LR
  subgraph Transformer Block
    X0[Input h_in] --> ATTN[Attention]
    ATTN --> FFN[FFN]
    FFN --> H1[h_blk]
  end
  H1 --> K[Key Proj]
  K --> CUBE[(Memory Cube)]
  H1 --> LNN[Liquid (LTC/CfC)]
  CUBE --> PRED[Residual Δy or Spike Recon]
  LNN --> LOUT[Liquid Output]
  PRED --> G{Gate α}
  LOUT --> G
  H1 --> G
  G --> Y[y_out]
```

HRM timing (slow/fast): T inner liquid steps per cycle; N cycles per segment; optional ACT halting across segments.

---

## 2. Working Assumptions

- States: fp32; params bf16/fp16; activations bf16. Residual clamps limit per-step Δ.
- Cubes store residual targets Δy = y_T − h_in by default (or projected variants).
- Gate α ∈ [0,1] with rate limit |Δα| ≤ ρ and hysteresis (α_up > α_down).
- Spike comb can encode selected features into sparse spikes and reconstruct near-perfectly at the block interface.

---

## 3. Formalism (Unified)

Let block boundary input be h ∈ R^d.

Teacher mapping: y_T = F_T(h; θ_T)

Key projection: k = φ(h) ∈ R^{d_k}

Cube retrieval (dense): (Δŷ, c_dense) = C_dense.query(k)

Spike comb (optional):
- s = Enc_spike(h_harmonic)  # events
- (Δŷ_spike, c_spike) = C_spike.query(s)
- Optionally decode spikes to reconstruct features: ĥ ≈ Dec_spike(s)

Liquid step (fast): (z_L, y_L) = LNN(h, z_L; θ_LNN, t)

Gate features r = [LN(h), LN(Δŷ_mix), c_mix, novelty]

α = σ(MLP(r)); y = (1 − α) y_T + α · y_m,
where y_m is liquid or residual form (h + Δŷ_mix), and Δŷ_mix fuses dense/spike predictions.

Online write: add (k, Δy) and/or (s, Δy) to cubes with caps and eviction.

---

## 4. Spike Comb Bridge (Codec & Metrics)

Purpose: create a sparse, event-based representation of selected block features with near-perfect/controllable reconstruction.

### 4.1 Encoding (draft)

Inputs: harmonic/phase-aware features u ∈ R^{d_h} derived from h (e.g., via sinusoidal or learned harmonic projection). Encode as events E = {(ch_i, t_i, a_i, τ_i)} with:
- Channel ch_i ∈ {1..C}
- Time t_i (continuous or discretized)
- Amplitude a_i (optional)
- Duration/decay τ_i (optional)

TODO[spec]: exact event schema (time-to-first vs delta times; binning; amplitude range). Latency budget per token.

### 4.2 Decoding (draft)

Reconstruct target features or residuals via superposition of kernels κ (e.g., exponential or alpha function):

ĥ_c(t) = Σ_i 1[ch_i=c] · a_i · κ(t − t_i; τ_i)

Guarantee knob: choose C, kernel family, and event budget per token to reach desired SNR/PR.

### 4.3 Distances/Similarity for Spikes

- Victor–Purpura distance d_VP(q) with cost q for shifting spikes in time.
- van Rossum distance d_vR(τ) via exponential filtering.
- Fusion with cosine on dense harmonics: sim_fuse = w_s · exp(−d_vR) + w_d · cos(k, K).

TODO[spec]: choose q, τ, and fusion weights; support learnable weights.

---

## 5. Neuroplastic Fast‑Weights (Optional)

Low‑rank fast delta on L or a cube-calibrator head, Oja/BCM-style with decay.

Oja update (per step):

ΔW = η · y · x^T − η · (y^T y) · W, W ← clip(W + ΔW, −w_max, w_max)

BCM flavor: use activity threshold θ_M (sliding) in potentiation term.

Decay: W ← (1 − 1/τ_fw) W each step/segment.

Modulators: scale η by gate confidence, novelty, or ACT advantage.

Safety: cap per-step ||ΔW||_F, audit drift in y vs y_T, rollback on exceedance.

TODO[spec]: pick rule (Oja vs BCM vs STDP variant), rank, η, τ_fw, caps.

---

## 6. Gate and Hysteresis

Features: r = [LN(h), LN(Δŷ_mix), conf, novelty]

α_raw = σ(MLP(r)); α_t = clamp(α_{t−1} + clip(α_raw − α_{t−1}, −ρ, +ρ), α_min, α_max)

Promotion: if conf > c_up and recent_err < ε_up → α_max += δ_up (≤ 1)

Demotion: if conf < c_down or drift > ε_down → α_max −= δ_down

Drift audits: recompute teacher periodically on samples; if ||y − y_T|| > δ across P audits, demote and purge offending cube entries.

---

## 7. HRM Timing & ACT Halting (Segment Level)

Inner loop: T liquid steps per cycle with H fixed; then H updates once (cycle end). Repeat N cycles per segment.

ACT: Q-head on z_H predicts {halt, continue}; halt when (m ≥ M_min and Q_halt > Q_cont) or m = M_max. Train with BCE on TD targets; detach (z_H,z_L) between segments for 1-step gradient.

---

## 8. Safety & UMA

- Residual clamps: ||Δ||_∞ ≤ c (default c=1.0) on liquid/cube contributions.
- UMA watchdog: soft (shrink cubes) / hard (abort segment) thresholds.
- Cube policies: per-cube and global caps; LRU/TTL; diversity-aware eviction.

---

## 9. Minimal Pseudocode (Reference)

```python
def cube_gated_block(h_in, teacher_block, cube_dense, cube_spike, liquid, gate_mlp, train: bool):
    y_T = teacher_block(h_in)                       # teacher
    k = phi(h_in)                                   # dense key
    d_pred, d_conf = cube_dense.query(k)
    # spike route (optional)
    s = spike_enc(h_in)                             # events
    s_pred, s_conf = cube_spike.query(s)
    d_pred, s_pred = match_dims(d_pred, s_pred)     # projection if needed
    pred_mix, conf_mix = fuse_preds(d_pred, s_pred, d_conf, s_conf)
    # liquid
    y_L = liquid_step(h_in)
    feats = torch.cat([LN(h_in), LN(pred_mix), conf_mix], dim=-1)
    alpha = sigmoid(gate_mlp(feats)); alpha = rate_limit(alpha)
    y_m = h_in + pred_mix                           # residual form
    y = (1 - alpha) * y_T + alpha * y_m
    if train:
        delta = (y_T.detach() - h_in.detach())
        cube_dense.update(k, delta)
        cube_spike.update(s, delta)
    return y, {"alpha": alpha.mean(), "conf": conf_mix.mean()}
```

---

## 10. API Plan (stubs; will evolve)

```python
class SpikeEncoder(nn.Module):
    def forward(self, h: Tensor) -> List[Events]: ...   # events per batch item

class SpikeDecoder(nn.Module):
    def forward(self, events: List[Events]) -> Tensor: ...

class MemoryCubeDense(nn.Module):
    def query(self, k: Tensor) -> Tuple[Tensor, Tensor]: ...
    def update(self, k: Tensor, v: Tensor) -> None: ...

class MemoryCubeSpike(nn.Module):
    def query(self, events: List[Events]) -> Tuple[Tensor, Tensor]: ...
    def update(self, events: List[Events], v: Tensor) -> None: ...

class PlasticityLowRank(nn.Module):
    def step(self, x: Tensor, y: Tensor) -> None: ...   # Oja/BCM update with decay
```

---

## 11. Experiments Matrix (initial)

- Recon quality: spike comb SNR/PR vs latency & sparsity (events/token).
- Retrieval: dense vs spike vs fused keys → hit rate, α trajectory, drift audits.
- Plasticity: Oja/BCM on/off; rank/η/τ sweeps; stability boundaries.
- HRM scaling: T,N,M_max sweeps; returns on compute (Sudoku, Maze, ARC-like).

Metrics: MSE/CE, α_mean, conf_mean, cube size/hitrate, drift, UMA events, speed.

---

## 12. Open Specs To Confirm (blocking precise impl)

1) Spike comb event schema (time coding, amplitude, duration) and per-token latency budget.
2) Recon target and guarantee (SNR/PR thresholds; dims to preserve).
3) Spike distance choice & params (Victor–Purpura q, van Rossum τ) and fusion weights.
4) Plasticity rule (Oja/BCM/STDP), rank, η, τ_fw, per-step Δ caps.
5) Persistence policy for spike cubes (on-disk size, TTL, format).
6) Placement scope (per block vs per head) for first pass.

---

## 13. Source Pointers (context we’re aligning with)

- Kognitive docs: LTC abstract/notes; Purkinje analysis; harmonic embeddings; temporal attention; Kotlin PRD.
- Our internal refs: HRM formalism; memory cubes; liquid cell; stability; ACT.

---

## 14. TODO Log

- TODO[spec]: finalize spike comb schema (+ metric params).
- TODO[impl]: add `spike_comb.py` (encoder/decoder + distances), `memory_cube_spike.py` (mixed retrieval).
- TODO[impl]: plasticity plugin with Oja/BCM and audits.
- TODO[exp]: recon SNR sweep; cube hit-rate A/B; α hysteresis stress tests; UMA pressure tests.
- TODO[doc]: expand math appendix with full derivations and proofs (IFT/DEQ, gate bounds).

---

## 15. Incorporating “pytorch_blocky_ltc” (Blocky LTC, Standard LTC, Liquid CFC × xLSTM)

Source: /Volumes/stuff/Projects/solace/Kognitive/pytorch_blocky_ltc

### 15.1 BlockySigmoidSTE and BlockyLTCCell

- Discrete activation via bucketized thresholds and values (17 levels), with a staircase surrogate gradient (STE) in backward.
- Cell dynamics: explicit Euler update with learnable log-τ and log-g_leak; input and recurrent linear maps; residual-like form dh = (act − h·(1+τ·g_leak))/τ.
- Rationale: deterministic quantization for controllable precision/amplitude, simpler hyperparams; potential speed/robustness; explicit control over degradation along chains.

Integration plan:
- Offer BlockySigmoidSTE as an activation option inside liquid cell variants.
- Add quantized-mode toggles in cube-gated blocks for research A/B (continuous vs blocky).
- Explore synergy with spike comb: blocky levels may map naturally to event amplitudes.

Open questions:
- Optimal level count and spacing; learned vs fixed μ/σ; STE shape calibration.
- Where to clip; how blocky interacts with residual clamps and α gating.

### 15.2 StandardLTCCell baseline

- Tanh activation, same Euler update path and parameterization; serves as sanity baseline.

### 15.3 LiquidCFCxLSTMCell (+ RNN wrapper)

- Hybrid gates (i,f,o,g) from recurrent and optional input projections; cell state c_t; normalizer n_t encouraging gate-sum ≈ target; hidden dynamics use closed-form decay: h_t = e^{−dtλ}·h_prev + (1−e^{−dtλ})·(o_t·σ(c_t)).
- Hebbian update term for W_recurrent per step: hebbian ∝ h_t ⊗ h_prev scaled by i_t, applied per-step or post-sequence with decay.
- Safety: clipping activations/states and weights.

Integration plan:
- Provide LiquidCFCxLSTM as an alternative L-module in HRM (drop-in for our liquid cell), exposing λ, α_norm, target_sum_norm, and Hebbian params (η, decay).
- Gate coupling: feed normalized gate sums and Hebbian activity into α features for trust modulation.
- Plasticity unification: compare built-in Hebbian vs low-rank Oja plugin; bind both to audits and caps.

### 15.4 sMNIST training harness

- Sequential MNIST pipeline supports MPS; CLI switch among {blocky, standard, liquid}; logs timing, accuracy.

Integration plan:
- Create a minimal adapter to route our hybrid blocks into this harness for quick empirical A/B and regression checks.

### 15.5 Immediate TODOs (this repo)

- TODO[impl]: Add BlockySigmoidSTE option in `src/lnn_hrm/liquid_time_constant.py` (flag `blocky=True`, pluggable activation).
- TODO[impl]: Add LiquidCFCxLSTM variant under `src/lnn_hrm/liquid_cfc_xlstm.py` (port with clean API) and expose as L-module choice in HRM.
- TODO[exp]: Port sMNIST runner as `examples/train_smnist_hybrid.py` to compare {standard, blocky, liquid} and {no-cube vs cube}.
- TODO[doc]: Document quantized activations and Hebbian xLSTM dynamics; note interactions with spike comb.

---

## 16. Grand Tour Log (Things to keep)

Liquid + Comb
- Spike comb enables near‑perfect/controllable recon from sparse events; mixes cleanly with dense harmonic features.
- Dual‑key cubes: cosine on harmonics + Victor–Purpura/van Rossum on spikes with learned fusion.
- Event budget is a first‑class knob for UMA and latency; burst–pause timing maps naturally into comb channels.

Quantized “Blocky” LTC
- Deterministic 17‑level activation (STE) simplifies behavior and makes amplitude/precision decay explicit.
- Plays well with spike amplitudes (shared ladders), clarifies safety clamps and residual bounds.

Liquid CFC × xLSTM
- Closed‑form h‑update via exp(−λΔt) is stable and interpretable; gate‑sum normalizer keeps regimes sane.
- Hebbian W_recurrent updates give plasticity “for free” (with caps/decay) and rich telemetry.

HRM Loop
- One‑step DEQ gradient + deep supervision → O(1) memory; ACT for inference‑time scaling without CoT.
- Clean H/L separation (slow plan, fast execute); deterministic canon mode (fixed chunks, fp32 states).

Memory Cubes
- Residual deltas as values are numerically safe and auditable; hysteresis + rate‑limited α enables graceful takeover.
- UMA‑aware: caps, LRU/TTL, soft shrink; persistent shards are straightforward.

Plasticity
- Low‑rank Oja/BCM with decay complements Hebbian; modulate by confidence/novelty/advantage.
- Rollback + audits as safety rails; unify plasticity policies across liquid and cubes.

Temporal Attention (H‑module)
- Time embeddings/gates align with HRM cycles; good home for global aggregation of comb/plasticity stats.

Purkinje/Dendrites Inspiration
- Multiplexed rate + burst timing supports comb event schema and cube locality (per block/head).

Kotlin Actors (“Virtual Dendrites”)
- Cubes/gates as actors with typed messages; clean persistence/backpressure; mirrors Python modularity.

Immediate Wins
- sMNIST A/B: {standard, blocky, LiquidCFC×xLSTM} × {no‑cube, cube}; log α, conf, drift, UMA, Hebbian norms.
- Mixed‑key cubes prototype; xltop‑style TUI for live telemetry.

Be Careful
- Calibrate STE; cap Hebbian/fast‑weight deltas; conservative fusion for spike distances; TTL from the start.

---

## 17. Bright Ideas (Preserve for later)

- Gate as Safety Critic: train a tiny risk head to temper α via conformal calibration on residual errors; temperature‑scale α under distribution shift.
- Cube Distillation: offline replay compresses heavy cubes into tiny calibrator MLPs or codebooks; swap when UMA tightens.
- Adaptive Delta Quantization: learn per‑cube residual codebooks matching blocky levels; ties directly to spike amplitude ladders.
- Spike Kernel Learning: learn κ family (alpha/exponential mixtures) and per‑channel τ_i to maximize recon SNR under event budgets.
- Dual‑Rail Gating: separate α_blend (output mix) from α_plastic (update rate); the latter can stay high while the former stays conservative.
- Replay‑Aware Plasticity: small FIFO of recent (h,Δy) pairs per cube to stabilize Oja/BCM updates and prevent drift.
- UMA‑First Scheduler: pre‑shrink cubes and freeze plasticity before forward when pressure predicted high (avoid mid‑pass churn).
- Comb‑coded KV: store KV cache deltas as spike events for memory‑efficient long‑context replay.
- Global Rate Budget Controller: allocate α_max budgets across cubes via a simple optimizer balancing drift and UMA costs.
- Domain‑Specific Comb Metrics: pick VP/van‑Rossum params per modality (text/audio/vision/time‑series) and learn fusion.
- LSH‑Aided Spike Retrieval: locality‑sensitive hashing on spike event embeddings to keep retrieval sub‑linear.
- PR Probes: routinely measure participation ratio across H/L/comb spaces to track hierarchical dimensionality.

---

## 18. Compaction Strategy (when context must shrink)

- Canonical Spec: HRM × Liquid × Cubes × Comb × Plasticity as one concise section; detailed math in an appendix.
- Minimal API Sheet: modules, flags, telemetry names; point to code paths rather than inlining.
- Results Digest: tight A/B tables (accuracy, speed, drift) plus 3–4 key plots; keep raw logs elsewhere.
- Playbook: tuning ranges, audits, UMA responses, fail‑safes, and a short checklist for “safe deployment mode.”

---

## 19. SolaceCore (Kotlin) — Actor/Port Integration Plan

Source: /Volumes/stuff/Projects/SolaceCore/docs (Architectural_Deepdive.md; Architectural_Document_Solace_Core_Framework.md; components/*; STORAGE docs)

What maps directly:
- Actor System → runtime container for cubes, gates, liquid modules, and auditors as separate actors.
- Ports → typed channels for: HarmonicFeatures, SpikeEvents, ResidualDelta, GateDecision, AuditRequest/Result, UMAEvents, Metrics.
- Supervisor → manages hot‑plugging (swap cube calibrators, change fusion weights, switch L‑module type), lifecycle, and resource limits.
- Protocol Adapters / Conversion Rules → encode/decode between dense (harmonics), spike events, and storage formats; apply calibrations.
- Workflow Manager (planned) → orchestrate the inference pipeline stages as a graph (Encode → Retrieve → Liquid → Gate → Audit).

Message taxonomy (first pass):
- `HarmonicFeatures(chunks: FloatArray, ts: Long)`
- `SpikeEvents(events: List<Event>, ts: Long)`  // Event: (channel, t, amp, dur)
- `ResidualDelta(vec: FloatArray, ts: Long)`
- `GateDecision(alpha: Float, conf: Float, novelty: Float, ts: Long)`
- `AuditRequest(sampleIds: List<Long>)` / `AuditResult(drift: Float, offenders: List<KeyRef>)`
- `UMAEvent(level: Soft|Hard, bytes: Long)`
- `Metrics(fields: Map<String, Any>)`

Hot‑pluggable patterns we’ll exploit:
- Swap gate MLP / fusion weights without pausing actors; keep stateful cubes running.
- Replace cube storage backends (in‑memory ↔ persisted shards) on the fly via Port reconnection.
- Toggle L‑module (LiquidLTC ↔ LiquidCFC×xLSTM ↔ Blocky) per actor instance.

Persistence & storage:
- Neo4j (planned) for relationships: actors↔cubes↔datasets; provenance of residuals and audits.
- Kotlin‑native storage for cube shards (keys/values/spike events) with TTL and versioning; Supervisor handles rollovers.

Observability:
- Actor metrics: α_mean, conf_mean, drift stats, Hebbian norms, cube hit/evict, UMA pressure; expose via typed `Metrics` messages.
- Structured logs with correlation IDs per request/segment.

Deployment sketch:
- Supervisor (Ktor in Docker) manages worker pods hosting actors; Workflow manager wires pipelines.
- Cluster‑ready lanes: UMA events throttle producers; backpressure via Port buffering; timeouts for audits.

Next steps (SolaceCore side):
- Define message types and Port adapters (dense↔spike↔storage).
- Actor skeletons: `CubeActor`, `GateActor`, `LiquidActor`, `AuditActor`, `SupervisorActor`.
- Minimal workflow: `Encode → Retrieve → Liquid → Gate → (optional Audit) → Output`.

---

## 20. TuringTest Archive Findings (TUNA) — What to Preserve & How To Use

Source: /Volumes/stuff/Projects/turingtest/archive (TUNA architecture; phonological loop; tests of universality)

Key artifacts skimmed:
- `reports/architecture.md`: TUNA levels (ion channels → dendrites → firing patterns → cognitive systems); SHIFT/XOR primitives; theta cycle scheduling; calcium logic.
- `reports/phonological_loop_design.md`: five channel “gates” (HCN1, Cav3.2, Kv7, BK, Nav1.6/1.7) mapped to XOR/INTEGRATE/FILTER/COINCIDENCE/OUTPUT_XOR over a 6 Hz theta cycle.
- `test_turing_completeness.py`: “Rule‑of‑3 HCN1 Learner” with reversible forward/reverse steps, energy‑drain by phase φ=V mod 3, boolean/state machine/tape/halting/program composition tests.
- `scripts/*`: cochlear and wavelet tooling, GF(2) analyses, Purkinje processing — useful for comb kernels and audio pipelines.

What we’ll integrate:
1) Phase‑coded gating
   - Expose a phase feature φ (e.g., modulo classes over time or harmonic components) into gate inputs.
   - Optionally schedule sub‑windows within HRM cycles (theta‑like windows), aligning comb events and liquid steps.

2) Ion‑gate ↔ Blocky/Comb alignment
   - Map ion thresholds to blocky activation levels; treat spike amplitude ladders as “channel thresholds”.
   - Use FILTER/COINCIDENCE patterns as comb‑side predicates (e.g., event coincidence within Δt windows) to raise/lower α.

3) Energy budgets as safety signals
   - Track a simple “energy” counter analogous to ΔE(φ); let high drain demote α_max or freeze plasticity.
   - Surface energy in telemetry beside α/conf/drift.

4) Reversibility hooks
   - Implement a reversible mode for certain liquid/cube updates (keep pre‑image caches for K steps) to support audit/replay.

5) Universality micro‑bench
   - Port the boolean/state machine/memory tape tests as a “universality suite” for the hybrid (dense‑only vs spike‑only vs fused; liquid on/off).
   - Add SHIFT/XOR residual codebooks and check composability.

6) Audio/cochlear tie‑in (comb)
   - Leverage cochlear scripts to derive spike events from basilar membrane outputs; test comb distances on real phonology.

Concrete TODOs:
- TODO[impl]: add `phase_feature(time, harmonics) → {0,1,2}` and append to gate features; parametrize windows per HRM cycle.
- TODO[impl]: “energy budget” accumulator and α_max scheduler; per‑segment cap if energy over threshold.
- TODO[test]: universality tests in `examples/universality_suite.py` with metrics (depth, energy, drift) under toggles.
- TODO[exp]: audio comb demo using cochlear filterbank + spike encoding; measure recon SNR and cube hit‑rate.

---

## 21. TUNA Detailed Integration Notes (Extreme Detail)

This section captures concrete mappings from TUNA artifacts into our hybrid’s design, with explicit encodings, timings, and test specs so nothing is lost.

### 21.1 Five Ion “Gates” ↔ Comb/Gate/Blocky Mapping

| Channel | Threshold (approx) | TUNA Role | Hybrid Mapping |
|---|---:|---|---|
| HCN1 | −85 mV | XOR_STORE (phonological store gateway) | Treat as XOR/STORE predicate over comb events: if (new_input XOR current_store) then store; in practice, raise write‑probability and α_plastic when XOR parity flips; expose φ feature from current phase bucket (see 21.2). |
| Cav3.2 (T‑type Ca²⁺) | −70 mV | INTEGRATE (encode features, trigger plasticity) | Map to liquid update bias and plasticity η boost; interpret as local integration gate: when active, permit Oja/BCM updates above baseline; increase Δy write confidence. |
| Kv7 (M‑current) | −60 mV | FILTER_NOT (noise reject + motor plan) | Implement as comb side filter: down‑weight keys that co‑occur with “interferers”; contribute a negative feature to gate MLP; could serve as novelty threshold modulator. |
| BK (fast K⁺) | −45 mV | COINCIDENCE (bind phonemes within Δt) | Spike comb coincidence detector: if ≥2 channels spike within window τ_c, add a coincidence bit; treat as sequence binding; boosts α_blend transiently. |
| Nav1.6/1.7 (Na⁺) | −55 mV | OUTPUT_XOR (action potential gating) | Implement as output decision switch: when confidence high and demand present, lift α cap for the final blend; else keep conservative. |

Blocky alignment: quantized “levels” (17‑level ladder) serve as analogues of thresholds; choose codebook bins aligned to the above voltages; spike amplitude ladders share the same bins for tight coupling.

### 21.2 Theta Cycle Windows and Phase Feature φ

TUNA’s phonological loop allocates 5 ops across a 6 Hz cycle (167 ms) at ~72° steps: 0°,72°,144°,216°,288°. We will: 
- Define per‑segment subwindows W0..W4 with durations from the timeline (e.g., 33.4 ms per window; adjustable). 
- Compute a phase bucket φ ∈ {0,1,2} via one of:
  1) φ_time = ⌊(t mod 167 ms)/ (167/3)⌋  
  2) φ_harm = argmax over 3 phase bins of the dominant harmonic component in h (learned mapping)  
  3) φ_key = (hash(k) mod 3) for deterministic tie‑breaker
- Feed φ (one‑hot) into gate features; optionally gate cube writes by window (e.g., HCN1 writes only in W0, Cav3.2 in W1, etc.). 
- Provide schedules for {HCN1,Cav3.2,Kv7,BK,Nav} enabling/disabling their influence per window; schedule adjustable per domain.

### 21.3 Calcium “Machine Code” ↔ Energy Budget

- Maintain E (energy units) as a segment counter. At each step, drain ΔE(φ): {0,1,2} per Rule‑of‑3 mapping or learned; report in telemetry. 
- Safety controller: if E falls below E_min, freeze plasticity (η=0), cap α_max↓, and increase audit frequency; if far below hard threshold, bypass cubes (teacher‑only). 
- Option: calibrate ΔE drain from comb event counts/intensities (spike–derived proxy for calcium demand). 

### 21.4 SHIFT/XOR Primitives in Residual Codebooks

- Create residual codebooks for SHIFT and XOR:
  - SHIFT: circular shift and stride in selected residual subvectors; implement as fixed small sparse matrices applied to Δy before/after cube; used to emulate data movement primitives. 
  - XOR: bitwise over quantized residual codes (blocky bins); gate selects between direct residual and XORed residual when comb parity triggers. 
- Program composition: stage 1 computes code index; stage 2 applies codebook op; track correctness via audit. 

### 21.5 Universality Suite (Spec)

We will port the tests from `test_turing_completeness.py` into `examples/universality_suite.py`:
- Boolean Gates: encode A,B in base‑3 pattern; run hybrid with specific schedules; decode outputs from (depth, energy, drift) to boolean; verify truth tables. 
- State Machine: base‑3 encoded transitions; verify stepwise updates and depth patterns. 
- Memory Tape: encode tape as base‑3; measure depth/energy; attempt recon from phase sequences; confirm head movement simulation via additive encodings. 
- Program Composition: f(g(x)) by chaining residual codebooks; track total energy and depths. 
- Unbounded Computation: iterate V values up to large ranges; analyze phase‑sequence repeats and branching. 
- Halting Behavior: ensure termination under schedules and report distributions across residues modulo 3^n. 

Toggles per run: {dense only, spike only, fused}, {liquid on/off}, {plastic on/off}, {blocky on/off}, {phase gating on/off}. Metrics: depth, total energy, drift, α_mean, conf_mean, hit rate, Hebbian norms, UMA. 

### 21.6 Audio/Cochlear ↔ Comb

- Use cochlear filterbank script to produce basilar membrane outputs (L/R). 
- Extract phonological features → spike events (channel, time, amplitude, duration). 
- Test spike distances (Victor–Purpura q, van Rossum τ) and fusion with cosine; measure recon SNR and cube hit rates; evaluate W0..W4 scheduling alignment versus speech syllable timing. 

### 21.7 Reversibility Mode

- Keep a ring buffer of last K pre‑images for cubes/liquid; add `reverse_step` for liquid (best‑effort) and cube write undo. 
- Provide audit API: replay last S steps forward/backward to confirm reversibility invariant on bounded parts. 

### 21.8 Provenance & Storage (from RESEARCH_ORGANIZATION)

- Partition persistent shards by topic (calcium, CFC, theta, phonology). 
- Store codebook versions and Δy provenance with topic tags and window/time info. 
- Keep extraction indices for ZIPs/CSVs/PNGs referenced in TUNA; link to cube entries via metadata. 

### 21.9 Edge Spec: Parameter Defaults

- Windows: W0..W4 = 33.4 ms each (167 ms total), adjustable per domain. 
- Phase: φ_time default; φ_harm fallback if time unavailable. 
- Energy: E0 per segment = 1000 (like HCN1 learner), drains {0,1,2}; E_min_soft=200; E_min_hard=50. 
- Spike distances: VP q=1.0; vR τ=10 ms; fusion weights w_s=0.5,w_d=0.5 (learnable). 
- Codebooks: 16‑bin blocky ladder; SHIFT granularity 2–4 positions; XOR over bin codes. 

---

## 22. SolaceCore vs. Hierarchical Reasoning Model (Alignment & Extensions)

Goal: Blaze our own trail while staying interoperable with the HRM conceptual core (slow/fast loops, deep supervision, 1‑step gradients, ACT). No “disable” path — SolaceCore runs with our enhancements on by default.

SolaceCore “Canonical Trail” (default on)
- fI: harmonic+dense encoders; optional cochlear front for audio.
- fL (fast): Liquid CfC/LTC (or LiquidCFC×xLSTM) with T inner steps; blocky activation optional but recommended for reproducibility; phase windows (W0..W4) active.
- fH (slow): temporal‑aware planner (xLSTM or Transformer blocks) updated every T; aggregates comb/plasticity/energy signals.
- Memory Cubes: residual Δy caches with fused keys (cosine + spike distances), confidence‑gated α with hysteresis; audits + persistence.
- Spike Comb: event codec on selected features; distances (VP/vR) fused with cosine; coincidence features feed the gate.
- Plasticity: low‑rank Oja/BCM (decay+caps) and/or Hebbian in LiquidCFC×xLSTM; α_plastic decoupled from α_blend.
- Energy Budget: calcium‑inspired drain ΔE(φ) modulates α caps, plasticity, and audit cadence.
- Training: deep supervision over segments (detach z), 1‑step gradient on last H/L steps, ACT halting with Mmin/Mmax.

Conceptual Alignment (HRM ↔ SolaceCore)
- Two‑timescale loop: T inner L‑steps per H update; cycles × segments exactly as HRM.
- 1‑step gradient + deep supervision: identical mechanics, O(1) memory.
- ACT: same Mmin/Mmax policy with Q‑head; our extra signals (conf/energy) can decorate features but not control termination logic directly.
- Latent reasoning: no CoT requirement; computation in hidden states. Cubes/liquid accelerate latent algorithms; α→small approximates pure HRM behavior without hard switches.

New Claims To Demonstrate (beyond HRM)
- Data efficiency: With cubes+comb+blocky on, fewer samples reach equal/greater accuracy on Sudoku/Maze; report sample‑efficiency curves.
- Safe compute under pressure: Energy‑budget controller preserves correctness (drift ≤ δ) under UMA constraints; show graceful α demotion and plasticity freeze.
- Key fusion wins: Fused keys (dense+spike) improve hit‑rate and α confidence without increasing drift; ablate to dense‑only / spike‑only.
- Quantized stability: Blocky lanes reduce variance in residuals and improve audit reversibility; show tighter drift/∆ norms.

Measurements That Mirror HRM Signatures
- Hierarchical convergence: forward residuals — L spikes/reset within cycles; H drifts slowly across cycles.
- PR hierarchy: PR(zH) >> PR(zL); PR(zH) scales with task diversity.
- Inference‑time scaling: accuracy vs Mmax (strong on Sudoku/Maze; modest on ARC‑like), with cubes/liquid on.

Config Surface (SolaceCore defaults in parentheses)
- T inner steps (T=4), N cycles/segment (N=2), Mmax (Mmax=4), ε for Mmin (ε=0.1).
- Liquid type (CfC|LTC|CFC×xLSTM = CfC), blocky (on), phase windows (on), energy budget (on).
- Cubes (on): top‑K=8–16, size=16–64k; fused distances (w_s=w_d=0.5 learnable).
- Plasticity (on): Oja rank=8, η=1e‑3, decay τ_fw=2k, per‑step ∥ΔW∥ cap.

Experiment Matrix (SolaceCore)
- Benchmarks: sMNIST → Sudoku‑Extreme → Maze‑Hard → ARC‑style grids.
- Curves: accuracy vs Mmax; accuracy vs samples; α/conf trajectories; drift histograms; PR(zH) vs PR(zL); UMA events.
- Ablations: {comb on/off, fused vs dense vs spike}, {blocky on/off}, {plastic on/off}, {energy budget on/off}, {Liquid CfC vs CFC×xLSTM}.

Artifacts To Produce
- Plots: residual profiles (per step), PR bars, compute‑used distributions, α/conf time series, energy vs α caps, drift CDFs.
- Tables: sample efficiency, UMA incidents, audit failures (zero‑target preferred), speed (tok/s). 

---

## 23. TuringTest “Processed” Integration (Formalisms & Biophysics)

Source: /Volumes/stuff/Projects/turingtest/processed (Formalisms, proofs, HCN tests)

What’s in there (key items):
- Formalisms.md: XOR‑3 space {-1,0,+1}³; ternary↔XOR‑3 mapping; 27‑tick spiral; ω = 2π/(27 t_P); XOR‑3 time step operation; Lambda‑neuron ↔ XOR‑3 equivalence; oscillation mapping (β,θ,δ across three “time dimensions”).
- mathematical_proof_summary.md: wave interference superposition; cross‑frequency beats; PAC equations; ion‑channel Boolean ops; 5‑stage theta pipeline (5 × 33.4 ms); half/full‑adder “assembly code”; speech mapping into oscillatory bands.
- neural_voltage_leak_as_binary_division_remainder.md: τ=20ms leak interpreted as quotient/remainder; EPSP as remainder compensation; timing bias.
- test_hcn_dendritic_comprehensive.py: biophysical HCN plots; membrane oscillations; serotonin modulation; JSON report with V½, f0, σ, I_max.

Tight mappings to SolaceCore:
- XOR‑3 / ternary: we already use mod‑3 phase φ and ternary code hints; define lightweight `xor3_feature(h)` (balanced ternary projection) to enrich gate inputs alongside φ.
- 27‑tick spiral: align with HRM segments × cycles grid; optionally use 27‑step diagnostic traces for residual/audit plotting; keep as a diagnostic story, not a hard constraint.
- Leak remainder: add “remainder feature” r(t)=1−exp(−t/τ) at τ=20ms per token/step; feed into gate to bias early decisions and to plasticity η scheduling.
- Biophysics constants: parameterize HCN1 (V½≈−85 mV, f0≈6 Hz, σ≈4 Hz, I_max) and wire as defaults for comb coincidence and phase windowing; HCN resonance can modulate spike enc/dec kernel weights when a “bio‑mode” flag is on.

Concrete hooks/specs:
- Gate features: concat [LN(h), LN(Δŷ), conf, novelty, φ_onehot, xor3_feature, remainder(t)]
- Plasticity: η(t) = η0 · (1 + κ · remainder(t)) capped; demote η when energy budget low.
- Comb coincidence: if two events within τ_c and channel pair ∈ learned table (BK‑like), set coincidence bit to raise α transiently.
- Serotonin demo: map neuromodulator → gain factor g(5‑HT) that scales comb write confidence and/or liquid I_max proxy; keep as optional perturbation path for stress tests.

Artifacts to add:
- `examples/xor3_features_demo.py`: show φ and xor3_feature influence on α/Δŷ over synthetic streams.
- `examples/hcn_bio_mode_demo.py`: use HCN constants to modulate spike kernel and phase windows; compare with neutral mode.
- `docs/lnn_hrm_hybrid/biophysics_params.md`: collect channel defaults and their usage points (gates, comb, plasticity, telemetry).
