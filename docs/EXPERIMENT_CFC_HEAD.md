# Experiment: CfC‑Style Head + Device Telemetry (from MetalCoroutinesTest)

Purpose
- Prototype a continuous‑time smoothing head (CfC) and a tiny device‑side telemetry ring in PyTorch/MPS, inspired by `NeuromorphicKernel.metal`.

What we mirror
- Exponential gates (i,f,o) with subtract‑n normalizer.
- Candidate update g via sigmoid; cell: c_new = f·c + i·g.
- CfC hidden update: h_new = (h_old + Δt·(o·sigmoid(c_new))) / (1 + Δt·λ).
- Optional gate/lambda masks; tiny int32 device ring for anomaly counts.

Where this lives
- Code: `mlstm_kernels/torch/experiments/cfc_head_experiment.py`
  - `cfc_head_step(...)` — pure ATen, fusable with `torch.compile`.
  - `make_telemetry_ring(device)` — int32[8] counter tensor on device.

Usage sketch
```python
import torch
from mlstm_kernels.torch.experiments.cfc_head_experiment import (
    cfc_head_step, CfcConfig, make_telemetry_ring, compiled_cfc_head
)

B, H, D = 1, 4, 64
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

dtype = torch.bfloat16
h = torch.zeros(B, H, D, device=device, dtype=dtype)
c = torch.zeros_like(h)
n = torch.zeros_like(h)
# preactivations (stand‑in: affine outputs)
pre_i = torch.randn(B, H, D, device=device, dtype=torch.float32)
pre_f = torch.randn_like(pre_i)
pre_o = torch.randn_like(pre_i)
pre_g = torch.randn_like(pre_i)

lam = torch.rand(B, H, D, device=device, dtype=torch.float32)
mask_g = torch.ones(B, H, D, device=device, dtype=torch.int)
mask_l = torch.ones_like(mask_g)
ring = make_telemetry_ring(device)

cfg = CfcConfig(alpha=1e-2, target_sum=3.0, clamp_logits=30.0)
step = compiled_cfc_head()
h, c, n, ff = step(h, c, n, pre_i, pre_f, pre_o, pre_g, lam, mask_g, mask_l, 0.01, cfg, ring)
print('ring[0]=', int(ring[0].item()))
```

Notes
- This experiment does not alter the canonical xLSTM path; it’s a separate module for exploration.
- For integration experiments:
  - Call `cfc_head_step` inside a small inner‑tile loop (`T_inner`) in the compiled step.
  - Keep canonical math as default; gate CfC under a flag.
- We intentionally avoid weight mutation/atomics in inference; the device ring provides a safe, tiny alternative for counters.

Next
- If desired, add a debug sanitization pass at tile boundaries (mask NaN/Inf and clamp) in drivers.
- Consider a tiny “fuse‑report” to print kernel counts around the CfC head under `TORCH_LOGS=+inductor`.

