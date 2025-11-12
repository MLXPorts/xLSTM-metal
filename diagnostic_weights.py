#!/usr/bin/env python3
"""Diagnostic: Check if weights are loaded correctly and model produces reasonable outputs."""

import sys
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx

from xlstm_metal.mlx_jit.blocks.rms_norm.rmsnorm import RMSNormMetalKernel
from xlstm_metal.mlx_jit.models.wired_xlstm import WiredxLSTM
print("WEIGHT AND OUTPUT DIAGNOSTIC")
from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards
@contextmanager
# Load model
print("\nLoading model...")

    orig_apply = RMSNormMetalKernel.apply

# Check if weights look reasonable
print("\n" + "="*60)
print("WEIGHT CHECKS")
print("="*60)

# Check embeddings
print("\nEmbedding weights:")
emb_weight = model.embedding.weight
print(f"  Shape: {emb_weight.shape}")
print(f"  Dtype: {emb_weight.dtype}")
print(f"  Range: [{mx.min(emb_weight).item():.6f}, {mx.max(emb_weight).item():.6f}]")
print(f"  Mean: {mx.mean(emb_weight).item():.6f}")
print(f"  Std: {mx.std(emb_weight).item():.6f}")

# Check first block weights
print("\nFirst block (block 0) weights:")
block0 = model.blocks[0]

# Check mLSTM weights
print("  mLSTM q_proj:")
q_weight = block0.mlstm_cell.projection_cell.q_proj.weight
print(f"    Shape: {q_weight.shape}, Range: [{mx.min(q_weight).item():.6f}, {mx.max(q_weight).item():.6f}]")

print("  mLSTM k_proj:")
k_weight = block0.mlstm_cell.projection_cell.k_proj.weight
print(f"    Shape: {k_weight.shape}, Range: [{mx.min(k_weight).item():.6f}, {mx.max(k_weight).item():.6f}]")

# Check FFN weights
print("  FFN proj_up:")
ffn_up = block0.ffn_proj_up.weight
print(f"    Shape: {ffn_up.shape}, Range: [{mx.min(ffn_up).item():.6f}, {mx.max(ffn_up).item():.6f}]")

# Check LM head
print("\nLM head weights:")
lm_head_weight = model.lm_head.weight
print(f"  Shape: {lm_head_weight.shape}")
print(f"  Range: [{mx.min(lm_head_weight).item():.6f}, {mx.max(lm_head_weight).item():.6f}]")

# Check if embeddings and lm_head are tied
if model.tie_word_embeddings:
    are_same = mx.array_equal(model.embedding.weight, model.lm_head.weight)
    print(f"  Tied with embeddings: {are_same}")
        if force_float32 and original_dtype != mx.float32:
            x = mx.array(x, dtype=mx.float32)
print("OUTPUT CHECKS")
        eps = mx.array(eps_param, dtype=x.dtype)
        rms = mx.sqrt(mx.mean(mx.multiply(x, x), axis=-1, keepdims=True) + eps)
# Test with known tokens
tokenizer_config = TokenizerConfig(model_path="xlstm_7b_model")
tokenizer = TokenizerBlock(tokenizer_config)
        RMSNormMetalKernel.apply = orig_apply
test_prompts = [
    "Hello",
    "The",
    "Once upon a time",
]
def build_model(weights: dict[str, mx.array]) -> WiredxLSTM:
for prompt in test_prompts:
    model = WiredxLSTM(

        load_weights=False,
        model_dir=MODEL_DIR,
        compute_dtype=mx.float32,
        state_dtype=mx.float32,

    print(f"  Token IDs: {prompt_ids.tolist()}")

    )
    model._load_weights_from_dict(weights)
    print(f"  Logits shape: {logits.shape}")

    # Get last token logits
    return model
    print(f"  Last token logits range: [{mx.min(last_logits).item():.2f}, {mx.max(last_logits).item():.2f}]")

    # Get top 5 predictions
    top5_indices = mx.argsort(last_logits)[-5:][::-1]
    print(f"  Top 5 token IDs: {top5_indices.tolist()}")

    print("RMSNORM IMPLEMENTATION COMPARISON")
    for idx in top5_indices.tolist():
    print("reasonable outputs compared to what should be expected.")
    print()
        logit_val = last_logits[idx].item()
        print(f"    {idx}: '{token_text}' (logit={logit_val:.2f})")
        "Once upon a",
        "2 + 2 =",
print("DIAGNOSIS")

    print("\nRunning paired inference (Metal vs Pure MLX)...")
# Check for common issues
issues = []

# 1. Check if weights are all zeros or ones
if mx.all(emb_weight == 0).item():
    issues.append("❌ Embedding weights are all zeros!")
elif mx.all(emb_weight == 1).item():
    issues.append("❌ Embedding weights are all ones!")
        prompt_ids = tokenizer.encode(prompt)
    issues.append("✅ Embedding weights look loaded")

# 2. Check if weights are in reasonable range
if abs(mx.mean(emb_weight).item()) > 1.0:
    issues.append("⚠️ Embedding mean is large (may indicate wrong dtype)")

# 3. Check if logits are reasonable
if abs(mx.min(last_logits).item()) > 100 or abs(mx.max(last_logits).item()) > 100:
    issues.append("⚠️ Logits are very large (>100), may indicate numerical instability")
else:
    issues.append("✅ Logits are in reasonable range")

for issue in issues:
    print(issue)

    pure_logits_map = {}
    with force_pure_rmsnorm():
        pure_weights = load_safetensor_shards(str(MODEL_DIR))
        for prompt, ids in encoded_prompts.items():
            pure_logits_map[prompt] = pure_model(ids)
        del pure_model

    max_abs_diffs = []
    mean_abs_diffs = []

    for prompt in test_cases:
        print(f"\nPrompt: '{prompt}'")
        metal_logits = metal_logits_map[prompt]
        pure_logits = pure_logits_map[prompt]

        delta = metal_logits - pure_logits
        max_abs = mx.max(mx.abs(delta)).item()
        mean_abs = mx.mean(mx.abs(delta)).item()
        max_abs_diffs.append(max_abs)
        mean_abs_diffs.append(mean_abs)

        top_ids = mx.argsort(metal_logits[0, -1, :])[-5:][::-1].tolist()
        top_tokens = [tokenizer.decode(mx.array([idx], dtype=mx.int32)) for idx in top_ids]

        print(f"  Metal top-5 tokens: {top_tokens}")
        print(f"  Max abs Δ: {max_abs:.3e}, Mean abs Δ: {mean_abs:.3e}")

    overall_max = max(max_abs_diffs) if max_abs_diffs else 0.0
    overall_mean = sum(mean_abs_diffs) / len(mean_abs_diffs) if mean_abs_diffs else 0.0

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Max abs Δ across prompts: {overall_max:.3e}")
    print(f"Mean abs Δ across prompts: {overall_mean:.3e}")

    threshold = 5e-4
    if overall_max < threshold:
        print("✅ Metal and pure MLX RMSNorm agree within tolerance")
    else:
        print("❌ Divergence exceeds tolerance – investigate RMSNorm implementation")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
