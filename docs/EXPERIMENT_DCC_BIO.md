# Experiment: Dendritic Comb Codec (Biological) — Tensorized

Source inspiration: `lambda_neuron/lambda_neuron/dcc_biological.py` (gold). This experiment ports the DCC algorithm into vectorized PyTorch so we can test:
- Weight/state encodings with perfect (or near‑float32) reconstruction.
- Event‑style sparsity (per‑level spike/excess) as a compressed representation.
- Potential boundary encodings for xLSTM state, keeping fused math in fp32 inside tiles.

Files
- `mlstm_kernels/torch/experiments/dcc_biological_experiment.py`
  - `dcc_encode_tensor(x, cfg)` → `(residue, carries)` where `carries` has an extra depth dim.
  - `dcc_decode_tensor(residue, carries, cfg)` → reconstruction.
  - `dcc_self_test()` quick check on device.

Algorithm (summary)
- For L levels with attenuation η and threshold τ:
  - At each level: if residue>τ, emit a “carry” (excess=residue−τ), clamp residue to τ, attenuate residue by η.
  - Decode uses: residue/η^L + Σ (excess_level / η^level).
- Biological mapping: threshold crossings → spikes; residue attenuation → dendritic cable; perfect reconstruction → synaptic integration.

Integration directions (not enabled by default)
- Offline weight encoding: store weights as (residue, carries); decode on load or tile boundaries.
- Boundary state encoding: at tile boundaries, encode (C,N) to reduce fp32 bandwidth; decode at tile start; keep step math in fp32.
- Event probes: log carry counts as a sparsity metric; could guide adaptive tiling in research settings.

Caveats
- Vectorized carry storage is dense (depth dimension) for simplicity; a real compressor would store sparse events.
- Default tolerance aims at float32 parity; the original scalar code demonstrates near machine‑precision reconstruction.

