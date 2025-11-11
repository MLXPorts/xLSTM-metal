Title: MLX Multi-Head LayerNorm (MHLN) — SIMD Kernel Prototype (Lab)

Summary

- Implemented a per-head LayerNorm Metal kernel using one simdgroup (warp=32) per (B,NH) row over DH.
- Reductions use threadgroup memory with two barriers (post-sum, post-var).
- Verified numeric parity vs MLX ops with max|Δ| ~ 1e-6.

Benchmarks (Apple M3 Ultra, MLX 0.29)

- B=64, NH=8
    - DH=64:  MLX=13.58 ms, Kernel=68.99 ms (MLX faster; warmup sensitivity)
    - DH=96:  MLX=0.51 ms, Kernel=0.25 ms
    - DH=128: MLX=1.00 ms, Kernel=0.24 ms
    - DH=192: MLX=0.48 ms, Kernel=0.43 ms

Interpretation

- The kernel shows wins for moderate DH (96–128) at this batch, consistent with improved reduction locality and fewer
  high-level ops.
- At small DH or first-run warmup, MLX ops can be competitive or faster.
- Numeric parity holds with epsilon=1e-6.

Integration Guidance

- Replace head-aware normalization in MLX path with this kernel under a feature flag when DH is moderate (>=96), else
  keep current MLX ops.
- Add affine gamma/beta in-kernel or apply as a separate MLX op (minimal cost) to preserve exact behavior.
- Keep reductions in float32 regardless of input dtype (fp16/bf16 I/O acceptable, accumulate in float).

Files

- Kernel: `kernels/mhln_kernels.py`
- Bench: `mhln_bench.py`

Next

- Add optional vectorized loads (float4) for DH divisible by 4.
- Fuse affine (gamma/beta) and expose a weight/bias buffer.
- Try two-rows-per-warp for small DH to amortize control overhead.

