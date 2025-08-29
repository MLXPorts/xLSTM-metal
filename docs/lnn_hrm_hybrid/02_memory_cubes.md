# Memory Cubes — Cosine top‑K with Confidence

`MemoryCube` implements a simple, fast content-addressable store:

- Retrieval: cosine similarity over normalized keys, top‑K, softmax at temperature.
- Confidence: 1 − (entropy / log K), used to gate blending.
- Fused keys: optional spike/comb encoder blended with dense keys.
- Policy: ring buffer (streaming); `reset()` to clear; `audit_snapshot()` for logs.

Interface:

```
value, conf, idx, scores = cube.query(q_key, spike_key=None, topk=8, temperature=0.1)
cube.update(key, value, spike_key=None)
```

Gated blend (`CubeGatedBlock`):

```
y = alpha(x, conf) * x + (1 - alpha(x, conf)) * (x + P(mem(q)))
```

Where `alpha` is a learned gate in [0,1], `q=Q(x)` is a pooled key, and `P` adapts the value to feature space.

