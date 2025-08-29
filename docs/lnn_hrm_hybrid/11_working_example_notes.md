# Working Example — Quick Notes

Run:

```
python examples/transformer_lnn_example.py
```

What you should see:
- Output tensor on `mps` (or CPU fallback).
- `alpha_mean` around ~0.45–0.55 for untrained weights.
- `conf` near 0.0 on first call, increasing on the second due to memory updates.

Why two calls?
- The example updates the cube after the first forward (in training mode),
  so the second call can actually retrieve something — confidence rises.

