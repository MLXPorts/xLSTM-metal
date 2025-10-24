# Register‑Efficient Reversible RNN (Pebbles) — Notes and Mapping

Source: `/Volumes/stuff/Projects/_workspace/research/pebbles/`
- Paper/notes: Register‑Efficient Reversible RNN.md (pebble game analogy; rematerialization)
- Code: `reversible-rnn-pytorch/` (standalone PyTorch implementation)

Core concepts (as implemented)
- Accumulator state h (additive update): invertible activation (softsign) and near‑identity recurrence (`W_h ≈ I`) to preserve past information; updates add pattern evidence.
- Envelope/gating e: a small detector on top of h extracting salient “events” (e.g., vowel/punctuation) via linear map + ReLU.
- Input‑dependent forgetting τ(x, h): learnable time constant via `tau = sigmoid(tau_base + tanh(w * (x−offset)/scale))`; punctuation drives higher τ (more forgetting), letters lower τ (more retention).
- Reversibility: use of invertible activation and additive form allows recomputation of intermediate states during training (pebble/rematerialization), reducing memory vs BPTT.

How this relates to our xLSTM work
- CfC kinship: their update `h ← (1−τ)·h + τ·h_update` mirrors the continuous‑time smoothing we prototyped (`h_new = (h + Δt·ff)/(1 + Δt·λ)`), where τ ≈ Δt·λ/(1+Δt·λ).
- Exponential gating: their envelope is ReLU; our xLSTM/sLSTM uses exponential gates and a normalizer — we can test an exponential‑sigmoid hybrid (exp gates for accumulation + sigmoid readout), keeping fusion intact.
- Register/tiling alignment: accumulator + envelope split matches our fused step (accumulate) and outer driver (detect/emit) separation; suggests clean inner‑tiling windows.
- Training vs inference: pebble rematerialization is a training‑time memory win; inference benefits mostly from the stable additive form and small states.

Experiment sketch (quarantined; not default)
- Implement a reversible‑flavored cell (ATen‑only) with:
  - `h_update = f_affine(h, x)`; `h ← (1−τ(x,h))·h + τ·softsign(h_update)`
  - `e = relu(W_env·h + b)`; output `W_out·e + b`
- Compare vs CfC step and sLSTM on a toy sequence (vowel/punct) using `torch.compile` on MPS.
- Track: fusion (kernel counts), throughput, and (if training) peak memory vs rematerialization.

Why document (and not merge) now
- Keeps our canonical xLSTM path stable; the reversible cell is an orthogonal research direction.
- Mapping to CfC gives a controlled A/B without destabilizing the mainline.

Pointers
- `reversible_rnn_pytorch/reversible_rnn_standalone.py`: end‑to‑end experiment and plots.
- `PAPER_IMPLEMENTATION_ANALYSIS.md`: decisions & formulas for τ, input representation, and device handling.

