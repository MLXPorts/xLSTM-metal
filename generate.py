#!/usr/bin/env python3
"""xLSTM Text Generation CLI – Command-Line Interface for Inference

Overview
--------
Command-line interface for text generation using xLSTM-Metal models.
Provides both single-shot generation and interactive chat modes with
configurable sampling parameters.

This script is the primary entry point for running pretrained xLSTM models
on Apple Silicon using MLX. It wraps the xLSTMRunner class with a user-
friendly CLI.

Features
--------
- **Single-shot generation**: Generate text from a single prompt
- **Interactive mode**: Continuous chat-style interaction
- **Model introspection**: Display model architecture and configuration
- **Wiring visualization**: ASCII/Unicode wiring diagrams for debugging
- **Flexible sampling**: Temperature, top-k, top-p (nucleus) sampling
- **HuggingFace integration**: Load models by path or HF model ID

Usage Patterns
--------------
Single generation:
  $ python generate.py --model xlstm_7b_model --prompt "Once upon a time" --max-tokens 100

Interactive chat:
  $ python generate.py --model NX-AI/xLSTM-7b --interactive

Model information:
  $ python generate.py --model xlstm_7b_model --info

Debug wiring:
  $ python generate.py --model xlstm_7b_model --prompt "Test" --show-wiring

Sampling Control
----------------
Temperature (--temperature, -t):
  Controls randomness. Lower = more deterministic, higher = more creative.
  - 0.0: Greedy decoding (always pick most likely token)
  - 0.7: Focused sampling (good for factual tasks)
  - 1.0: Default (balanced creativity)
  - 1.5+: High creativity (may produce nonsense)

Top-k (--top-k):
  Sample from k most likely tokens.
  - Common values: 40, 50
  - Limits vocabulary to most probable choices

Top-p / Nucleus (--top-p):
  Sample from smallest set of tokens with cumulative probability >= p.
  - 0.9: Standard (diverse but coherent)
  - 0.95: More diverse
  - Adaptive: vocabulary size varies per token

Interactive Mode
----------------
In interactive mode (--interactive, -i):
  - Continuous prompt-response loop
  - Each generation maintains no context (stateless)
  - Ctrl+C to exit
  - Empty prompts are skipped

For stateful conversation (context retention across turns), use the
runner API directly with state management.

Model Loading
-------------
The script accepts:
  1. Local path: --model /path/to/xlstm_7b_model
  2. HuggingFace ID: --model NX-AI/xLSTM-7b

Models must have:
  - config.json (model configuration)
  - model.safetensors.index.json (weight sharding info)
  - model-*.safetensors (weight shards)
  - tokenizer.json (tokenizer vocabulary and config)

Wiring Visualization
--------------------
The --show-wiring flag displays NCPS wiring structure:
  - ASCII: Basic characters (portable)
  - Unicode: Box-drawing characters (prettier)

Useful for debugging model architecture and connectivity patterns.

Output Format
-------------
Single-shot mode prints generated text directly to stdout.
Interactive mode uses "> " prompt and "Generated: " prefix.
Model info uses structured key-value output.

Error Handling
--------------
Common errors:
  - Model not found: Check path/HF ID, ensure downloaded
  - Out of memory: Reduce max-tokens or use smaller model
  - Tokenizer error: Ensure tokenizer.json present in model dir
  - Import error: Check xlstm_metal package installation

Performance Notes
-----------------
First run is slower (model loading + Metal shader compilation).
Subsequent generations are faster (cached shaders).
Long sequences (>2048 tokens) may require chunking for memory.

Examples
--------
Creative story generation:
  $ python generate.py -m xlstm_7b_model -p "In a distant galaxy" \\
      --max-tokens 200 --temperature 1.2 --top-p 0.95

Factual Q&A:
  $ python generate.py -m xlstm_7b_model -p "What is the capital of France?" \\
      --max-tokens 20 --temperature 0.3 --top-k 10

Interactive chat:
  $ python generate.py -m NX-AI/xLSTM-7b -i --max-tokens 100 --temperature 0.8

Parity
------
This CLI mirrors torch-native generate.py for cross-backend compatibility.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from xlstm_metal.mlx_jit.generate import xLSTMRunner
from xlstm_metal.mlx_jit.tokenizer import TokenizerBlock, TokenizerConfig


def main():
    """Main entry point for xLSTM text generation CLI.

    Parses command-line arguments, loads model and tokenizer, and runs
    generation in either single-shot or interactive mode.

    Command-Line Arguments
    ----------------------
    Required (one of):
      --prompt, -p : Input prompt text for single generation
      --interactive, -i : Enable interactive chat mode
      --info : Display model architecture information

    Model Configuration:
      --model, -m : Model path or HuggingFace ID (required)

    Generation Parameters:
      --max-tokens : Maximum tokens to generate (default: 50)
      --temperature : Sampling temperature (default: 1.0)
      --top-k : Top-k sampling cutoff (optional)
      --top-p : Nucleus sampling threshold (optional)

    Debug Options:
      --show-wiring : Print NCPS wiring diagram
      --wiring-style : Diagram style (unicode|ascii, default: unicode)
      --info : Show model configuration details

    Raises
    ------
    argparse.ArgumentError
        If no prompt, interactive, or info flag provided.
    FileNotFoundError
        If model directory or required files not found.
    ImportError
        If xlstm_metal package not properly installed.

    Examples
    --------
    >>> # (Run from command line, not Python REPL)
    >>> # python generate.py -m xlstm_7b_model -p "Hello" --max-tokens 20
    """
    parser = argparse.ArgumentParser(description="xLSTM Text Generation Runner")

    parser.add_argument("--model", "-m", required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--prompt", "-p", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--show-wiring", action="store_true",
                        help="Print wiring diagram (debug)")
    parser.add_argument("--wiring-style", choices=["unicode", "ascii"], default="unicode",
                        help="Diagram style when using --show-wiring")
    parser.add_argument("--info", action="store_true", help="Show model info")

    args = parser.parse_args()

    if not args.interactive and not args.prompt and not args.info:
        parser.error("Must provide --prompt, --interactive, or --info")

    # Load model
    print(f"Loading model: {args.model}")
    runner = xLSTMRunner(
        args.model,
        show_wiring=args.show_wiring,
        wiring_style=args.wiring_style,
    )

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
