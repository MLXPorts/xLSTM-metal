#!/usr/bin/env python
"""
Test kernel mathematics with detailed logging
"""

import mlx.core as mx
from pathlib import Path
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("Kernel Math Debug Test")
    print("="*80)

    model_dir = Path("xlstm_7b_model")

    # Load
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    runner = xLSTM7BRunner(
        embedding_dim=4096,
        num_heads=8,
        num_blocks=32,
        vocab_size=50304,
        output_logit_soft_cap=30.0
    )
    runner.load_weights(str(model_dir))
    print("✓ Model loaded\n")

    # Test with longer prompt
    prompt = "The capital of France is Paris. The capital of Germany is"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Input IDs ({len(input_ids)} tokens): {input_ids}\n")

    # Forward pass
    runner.reset_state()
    current_ids = mx.array([input_ids])

    print("Running forward pass...")
    logits, runner.state = runner.forward(current_ids, runner.state)
    mx.eval(logits)

    # Check logits
    print(f"Logits shape: {logits.shape}")
    next_token_logits = logits[0, -1, :]

    print(f"\nLogits statistics:")
    print(f"  Min: {mx.min(next_token_logits).item():.4f}")
    print(f"  Max: {mx.max(next_token_logits).item():.4f}")
    print(f"  Mean: {mx.mean(next_token_logits).item():.4f}")
    print(f"  Std: {mx.std(next_token_logits).item():.4f}")

    # Check for NaN or Inf
    has_nan = mx.any(mx.isnan(next_token_logits)).item()
    has_inf = mx.any(mx.isinf(next_token_logits)).item()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    # Top 10 tokens
    top_k = 10
    top_indices = mx.argsort(next_token_logits)[-top_k:][::-1]

    print(f"\nTop {top_k} tokens:")
    for i in range(top_k):
        token_id = top_indices[i].item()
        logit_value = next_token_logits[token_id].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id:5d} ('{token_text}'): logit = {logit_value:.4f}")

    # Check expected token
    expected_tokens = ["Berlin", " Berlin"]
    print(f"\nExpected tokens and their logits:")
    for exp_text in expected_tokens:
        exp_ids = tokenizer.encode(exp_text, add_special_tokens=False)
        for exp_id in exp_ids:
            exp_logit = next_token_logits[exp_id].item()
            print(f"  Token {exp_id:5d} ('{tokenizer.decode([exp_id])}'): logit = {exp_logit:.4f}")

    # Generate next token
    next_token = mx.argmax(next_token_logits).item()
    print(f"\nArgmax next token: {next_token} ('{tokenizer.decode([next_token])}')")

    # Now test stateful generation
    print("\n" + "="*80)
    print("Testing stateful generation (generate 5 tokens)")
    print("="*80)

    generated = list(input_ids)
    current_ids = mx.array([[next_token]])

    for step in range(5):
        logits, runner.state = runner.forward(current_ids, runner.state)
        mx.eval(logits)

        next_token_logits = logits[0, -1, :]
        next_token = mx.argmax(next_token_logits).item()
        generated.append(next_token)

        print(f"Step {step+1}: Generated token {next_token} ('{tokenizer.decode([next_token])}')")
        print(f"  Logit range: {mx.max(next_token_logits).item() - mx.min(next_token_logits).item():.4f}")

        current_ids = mx.array([[next_token]])

    print(f"\nFull generated text:")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
