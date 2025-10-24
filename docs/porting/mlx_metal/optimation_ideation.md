Title: xLSTM MLX/PyTorch Improvement Plan

Goals
- Improve MLX xLSTM numerical stability and performance on Apple GPUs.
- Reduce kernel launches and memory churn in critical inner loops.
- Align MLX numerics and behavior with the stronger PyTorch implementation.
- Keep changes surgical and behind flags where risky.

Non-Goals
- No wholesale refactor or unrelated feature work.
- No external dependencies or network downloads.

Baseline References
- docs/mlx_reference/Comprehensive-MLX-Metal-Guide.md
- docs/mlx_reference/WWDC16-Optimization-Patterns.md (WWDC16 Session 606 patterns)
- mlx_fast_kernels/gemm_kernels.py, qr_kernels.py, shaders.py (working MLX Metal kernels)
- tools/mlx_runtime.py and tools/mlx_tuning.py (runtime knobs and device-aware tile defaults)
- tests/test_xlstm_mlx_inference_parity.py (fast head parity)

Summary Of Opportunities (MLX)
1) BlockLinear inefficiency (implementations/mlx/xlstm_mlx.py)
   - Today: constructs a full block-diagonal matrix per call (block_diag) and multiplies once.
   - Issue: High memory traffic (O((H·D)^2)) and unnecessary work; drives large matmul for per-head ops.
   - Plan: Replace with HeadLinear that keeps per-head weights (H, Do, Di) and computes outputs per-head without building a block matrix. Prefer vectorized/batched matmul or small per-head matmuls if batching is unavailable. Register arrays on the module for training.

2) Head-aware normalization
   - Today: GroupNorm over flattened hidden_dim (NH·DH) is used directly, which is not head-aware to the degree the PyTorch MultiHeadLayerNorm expects.
   - Plan: Implement an MLX MultiHeadLayerNorm that reshapes (B, NH, DH), normalizes per head in float32, then flattens and applies (optional) gamma/beta. Replace GroupNorm in sLSTM/mLSTM blocks.

3) Soft caps (numerical stability)
   - Today: MLX i/f gates and final logits do not apply soft caps; PyTorch does.
   - Plan: Apply gate soft cap: cap * tanh(x/cap) with cap=15.0 to i/f in sLSTM and mLSTM; apply logit soft cap with cap=30.0 to final logits. Add helpers; consider MLX Metal soft-cap kernel for throughput if beneficial (mlx_fast_kernels/shaders.py).

4) Sequence-level projection precompute
   - Today: For each timestep, several Linear ops execute, causing many kernel launches per token.
   - Plan: Precompute all W_* projections over the full sequence per block once (B,S,…) and scan the recurrent state. Reduces dispatch overhead and improves throughput.

5) Fast head (tiled GEMM) control
   - Today: self.use_fast_head = True by default; training can stumble on custom-kernel autograd paths.
   - Plan: Default fast head OFF for training; enable via tools.mlx_runtime.configure_model(fast_head=True) or set_fast_head(True) for inference. Keep robust fallback to nn.Linear on any kernel error.

6) Mixed precision & streams (optional, gated)
   - Add lightweight knobs for param_dtype/compute_dtype; keep accumulations in fp32 for reductions. Consider placing prefill/decode on a dedicated MLX stream (tools/mlx_streams.py) when driving pipelines.

Proposed Code Changes (Surgical)
- File: implementations/mlx/xlstm_mlx.py
  1) Replace BlockLinear
     - Add HeadLinear module with per-head weights: weight ∈ (H, Do, Di), optional bias ∈ (H, Do).
     - Update sLSTMBlock.R_{z,i,o,f} and mLSTMBlock projections to use HeadLinear.
     - Remove runtime block_diag assembly; keep block_diag helper only for tests.

  2) MultiHeadLayerNorm (MLX)
     - Add class MultiHeadLayerNorm: reshape to (B, NH, DH), compute mean/var in float32 along DH, scale/bias, return flattened (B, NH·DH).
     - Replace nn.GroupNorm usage in sLSTMBlock/mLSTMBlock with this layer.

  3) Soft caps
     - Add soft_cap(x, cap=15.0) and apply to i/f gates: i_t = soft_cap(W_i + R_i), f_t = soft_cap(W_f + R_f) in sLSTM; i_t = soft_cap(W_i), f_t = soft_cap(W_f) in mLSTM.
     - Cap final logits with soft_cap(..., 30.0) before returning.

  4) Sequence precompute
     - Precompute W_* for each block over (B,S,…) once, then time-scan recurrent states.
     - Guard behind a flag (e.g., self.sequence_precompute=True) and fall back for small S.

  5) Fast head guard
     - self.use_fast_head default False; if training=True or autograd active, force False. Only enable when explicitly configured (tools.mlx_runtime or set_fast_head).

  6) Dtype & stream hooks (behind flags)
     - Accept optional param_dtype/compute_dtype; cast compute-heavy paths; accumulate in fp32.
     - Accept optional stream for prefill/decode; default None.

- File: implementations/pytorch/xlstm_pytorch.py
  - Ensure MultiHeadLayerNorm is available: either import from xlstm_pytorch_enhanced or inline a minimal implementation to match usage.
  - Prefer the enhanced implementation path for consistency (or update root shim’s import target accordingly).

Validation & Benchmarks
- Shape/Decode Parity: tests/test_xlstm_mlx_inference_parity.py already verifies fast head parity; keep it green after changes.
- New atomic lab tests (see lab/ below) to prove:
  - HeadLinear vs block-diag correctness and reduced overhead.
  - MultiHeadLayerNorm behavior and stability vs naive GroupNorm.
  - Soft-cap numeric parity vs pure-MLX vs Metal kernel variant (mlx_fast_kernels/shaders.soft_cap).
  - GEMM tile sizing sweeps (16x16, 32x8, 8x32, 16x8, 8x16) for device-aware choices.
  - Sequence projection precompute vs per-step dispatch (time and throughput).
  - Stream overlap demo for decode/prefill.

Risks & Mitigations
- Batched per-head matmul support in MLX is limited: where not available, H small matmuls may increase launch count. Mitigate by (a) fusing across S first, (b) choosing thresholds where block-diag avoids huge weights but keeps launch count reasonable, (c) revisit with a custom Metal kernel if needed.
- Head-aware LN numerics: normalize per head in float32; provide epsilon and scale/bias; back out to GroupNorm via a flag if needed.
- Fast head in training: keep default OFF with safe fallback to nn.Linear.

Milestones
1) Land lab/ atomic tests and plan document (this file).
2) Implement HeadLinear + soft caps + MLX MultiHeadLayerNorm; keep flags off by default; validate with lab and tests.
3) Add sequence precompute flag and benchmark wins on medium S.
4) Integrate training defaults (fast head off; dtype hooks optional).
5) Optional: import fix for PyTorch MultiHeadLayerNorm / point shim to enhanced module.

Appendix: WWDC Patterns Applied
- Use 16-bit types where safe; keep accumulators in float (fp32) (WWDC data type guidance).
- Avoid dynamically indexed local arrays; prefer TG staging with fixed-size tiles.
- Use fma in inner loops; coalesced loads; two barriers per tile (post‑load, post‑accumulate).
- Favor 16×16 as safe tile size; evaluate 32×8 and 8×32 per device (threadExecutionWidth≈32); also test 16×8/8×16 variants noted in our experiments.

