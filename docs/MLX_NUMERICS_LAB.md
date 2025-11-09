# Experiments / Lab

This folder contains investigation utilities and traces used for numerical analysis and parity:

- `trace_hyena_divergence.py`, `trace_monarch_divergence.py` — step-by-step tracers for MLX vs Torch.
- `debug_hyena_filter_trace.py` — focused Hyena stage tracer.
- `naive_hyena_conv_ref.py` — float32 per-op rounded reference for time-domain conv.
- `gpt5_details.txt` — investigation notes.

Profiles to stabilize numerics live in `bert/profiles/` (e.g., `torch_like.json`, `mlx_stable.json`, `torch_avg.json`).
Use the HyperProfile loader in `bert/src/mm_mlx/hyperprofiles_mlx.py` to select a profile.

These scripts are not part of production paths and may change; use them to diagnose numeric differences.
