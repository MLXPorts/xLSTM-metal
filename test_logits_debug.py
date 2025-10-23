#!/usr/bin/env python
"""
Debug logits to see why model only generates token 0
"""

import mlx.core as mx
from pathlib import Path
from xlstm_metal.inference.xlstm_7b_runner import xLSTM7BRunner
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("Logits Debug Test")
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

    # Simple prompt
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Input IDs: {input_ids}\n")

    # Forward pass
    runner.reset_state()
    current_ids = mx.array([input_ids])
    
    print("Running forward pass...")
    logits, runner.state = runner.forward(current_ids, runner.state)
    mx.eval(logits)
    
    # Check logits
    print(f"Logits shape: {logits.shape}")
    next_token_logits = logits[0, -1, :]  # [vocab_size]
    
    print(f"\nLogits statistics:")
    print(f"  Min: {mx.min(next_token_logits).item():.4f}")
    print(f"  Max: {mx.max(next_token_logits).item():.4f}")
    print(f"  Mean: {mx.mean(next_token_logits).item():.4f}")
    print(f"  Std: {mx.std(next_token_logits).item():.4f}")
    
    # Top 10 logits
    top_k = 10
    top_values = mx.topk(next_token_logits, k=top_k)
    top_indices = mx.argsort(next_token_logits)[-top_k:][::-1]
    
    print(f"\nTop {top_k} tokens:")
    for i in range(top_k):
        token_id = top_indices[i].item()
        logit_value = next_token_logits[token_id].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id:5d} ('{token_text}'): logit = {logit_value:.4f}")
    
    # Check if all logits are similar (indicating a problem)
    logit_range = mx.max(next_token_logits).item() - mx.min(next_token_logits).item()
    print(f"\nLogit range: {logit_range:.4f}")
    if logit_range < 0.01:
        print("⚠️  WARNING: Logits have very small range - model may not be working correctly!")
    
    # Sample with argmax
    next_token = mx.argmax(next_token_logits).item()
    print(f"\nArgmax next token: {next_token} ('{tokenizer.decode([next_token])}')")


if __name__ == "__main__":
    main()
