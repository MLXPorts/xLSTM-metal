#!/usr/bin/env python3
"""Diagnostic: Compare Metal RMSNorm vs pure-MLX fallback numerically."""

import sys
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx

from xlstm_metal.mlx_jit.blocks.rms_norm.rmsnorm import RMSNormMetalKernel
from xlstm_metal.mlx_jit.models.wired_xlstm import WiredxLSTM
from xlstm_metal.mlx_jit.tokenizer import TokenizerBlock, TokenizerConfig


@contextmanager
def force_pure_rmsnorm():
    """Temporarily replace the Metal kernel with a pure-MLX reference."""

    orig_apply = RMSNormMetalKernel.apply

    def pure_apply(self, inputs_2d, weight, eps_param, force_float32):
        x = inputs_2d
        original_dtype = x.dtype
        if force_float32 and original_dtype != mx.float32:
            x = mx.array(x, dtype=mx.float32)

        eps = mx.array(eps_param, dtype=x.dtype)
        rms = mx.sqrt(mx.mean(mx.multiply(x, x), axis=-1, keepdims=True) + eps)
        y = mx.multiply(x / rms, weight.astype(x.dtype))
        return y.astype(original_dtype)

    RMSNormMetalKernel.apply = pure_apply
    try:
        yield
    finally:
        RMSNormMetalKernel.apply = orig_apply

def main() -> None:
    print("="*60)
    print("RMSNORM IMPLEMENTATION COMPARISON")
    print("="*60)
    print("\nThis test checks if our Pure MLX RMSNorm produces")
    print("reasonable outputs compared to what should be expected.")
    print()

    print("Loading baseline (Metal RMSNorm) model...")
    metal_model = WiredxLSTM.from_pretrained(
        'xlstm_7b_model',
        compute_dtype=mx.float32,
        state_dtype=mx.float32,
        norm_reduce_force_float32=True,
    )
    print("✓ Baseline model loaded")

    tokenizer_config = TokenizerConfig(model_path="xlstm_7b_model")
    tokenizer = TokenizerBlock(tokenizer_config)

    print("\n" + "="*60)
    print("PREDICTION QUALITY CHECK")
    print("="*60)

    test_cases = [
        "Hello",
        "The quick brown",
        "Once upon a",
        "2 + 2 =",
    ]

    print("\nRunning paired inference (Metal vs Pure MLX)...")

    encoded_prompts = {}
    for prompt in test_cases:
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids.ndim == 1:
            prompt_ids = mx.expand_dims(prompt_ids, axis=0)
        encoded_prompts[prompt] = prompt_ids

    metal_logits_map = {}
    for prompt, ids in encoded_prompts.items():
        metal_logits_map[prompt] = metal_model(ids)

    pure_logits_map = {}
    with force_pure_rmsnorm():
        pure_model = WiredxLSTM.from_pretrained(
            'xlstm_7b_model',
            compute_dtype=mx.float32,
            state_dtype=mx.float32,
            norm_reduce_force_float32=True,
        )
        for prompt, ids in encoded_prompts.items():
            pure_logits_map[prompt] = pure_model(ids)

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
