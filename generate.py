#!/usr/bin/env python3
"""
xLSTM Text Generation Runner

Simple command-line interface for the xLSTM-Metal implementation.
Calls the existing xlstm_metal.inference.generate module.

Usage:
    python generate.py --model NX-AI/xLSTM-7b --prompt "Hello world" --max-tokens 50
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.mlx_jit.generate import xLSTMRunner
from xlstm_metal.mlx_jit.blocks.tokenizer.block import TokenizerBlock, TokenizerConfig


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser(description="xLSTM Text Generation Runner")
    
    parser.add_argument("--model", "-m", required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--prompt", "-p", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--info", action="store_true", help="Show model info")
    
    args = parser.parse_args()
    
    if not args.interactive and not args.prompt and not args.info:
        parser.error("Must provide --prompt, --interactive, or --info")
    
    # Load model
    print(f"Loading model: {args.model}")
    runner = xLSTMRunner(args.model)
    
    # Initialize tokenizer
    tokenizer_config = TokenizerConfig(model_path=args.model)
    tokenizer = TokenizerBlock(tokenizer_config)
    
    if args.info:
        info = runner.get_model_info()
        print(f"\nModel info:")
        print(f"  Blocks: {info['num_blocks']}")
        print(f"  Embedding dim: {info['embedding_dim']}")
        print(f"  Vocab size: {info['vocab_size']}")
        print(f"  Heads: {info['num_heads']}")
        print(f"  Chunk size: {info['chunk_size']}")
        return
    
    if args.interactive:
        print("Interactive mode (Ctrl+C to exit)")
        while True:
            prompt = input("\n> ").strip()
            if not prompt:
                continue
                
            prompt_ids = tokenizer.encode(prompt).tolist()
            generated_ids = runner.generate(
                prompt_ids, 
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            output = tokenizer.decode(generated_ids)
            print(f"Generated: {output}")
    else:
        # Single generation
        prompt_ids = tokenizer.encode(args.prompt).tolist()
        generated_ids = runner.generate(
            prompt_ids,
            max_tokens=args.max_tokens, 
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        output = tokenizer.decode(generated_ids)
        print(output)


if __name__ == "__main__":
    main()