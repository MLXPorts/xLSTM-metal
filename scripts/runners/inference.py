
"""
xLSTM Inference CLI

Usage:
    python inference.py "Your prompt here"
    python inference.py "Your prompt" --max_length 100 --temperature 0.7
    python inference.py "Your prompt" --model_size small
    python inference.py "Your prompt" --model_path /path/to/checkpoint.pt
"""

import argparse
import sys
import os
import time
import torch
from pathlib import Path

# Add xlstm to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xlstm_impl.models.xlstm import xLSTMLarge, xLSTMLargeConfig
from xlstm_impl.utils.device import DEVICE, get_device_info

class SimpleTokenizer:
    """Simple character-level tokenizer for testing."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Build character mappings
        for i in range(min(256, vocab_size)):
            char = chr(i)
            self.char_to_id[char] = i
            self.id_to_char[i] = char
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.unk_token_id)
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """Convert token IDs to text."""
        chars = []
        for id in ids:
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])
            else:
                chars.append('?')
        return ''.join(chars)

def get_model_config(size: str = "small") -> xLSTMLargeConfig:
    """Get model configuration by size."""
    configs = {
        "tiny": xLSTMLargeConfig(
            embedding_dim=128,
            num_heads=2,
            num_blocks=2,
            vocab_size=256,
        ),
        "small": xLSTMLargeConfig(
            embedding_dim=256,
            num_heads=4,
            num_blocks=4,
            vocab_size=256,
        ),
        "medium": xLSTMLargeConfig(
            embedding_dim=512,
            num_heads=8,
            num_blocks=8,
            vocab_size=256,
        ),
        "large": xLSTMLargeConfig(
            embedding_dim=1024,
            num_heads=16,
            num_blocks=12,
            vocab_size=256,
        ),
    }
    
    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from: {list(configs.keys())}")
    
    return configs[size]

def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> int:
    """Sample next token from logits with temperature and top-k filtering."""
    if temperature == 0:
        # Greedy sampling
        return logits.argmax(-1).item()
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)
    
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Sample
    return torch.multinomial(probs, num_samples=1).item()

def generate_text(
    model: xLSTMLarge,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu"
) -> str:
    """Generate text from a prompt."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_ids = input_ids.copy()
    state = None
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...")
    print("-" * 50)
    
    # Generate tokens
    with torch.no_grad():
        for i in range(max_length):
            # Get model predictions
            if state is None:
                # Process entire sequence
                logits = model(input_tensor)
                if isinstance(logits, tuple):
                    logits, state = logits
                next_token_logits = logits[0, -1, :]
            else:
                # Process only the last token with state
                last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
                logits, state = model(last_token, state=state)
                next_token_logits = logits[0, -1, :]
            
            # Sample next token
            next_token_id = sample_from_logits(next_token_logits, temperature, top_k)
            generated_ids.append(next_token_id)
            
            # Print generated token
            token_text = tokenizer.decode([next_token_id])
            print(token_text, end='', flush=True)
            
            # Update input for next iteration
            if state is None:
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
    
    print("\n" + "-" * 50)
    
    # Decode full text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="xLSTM Inference CLI")
    parser.add_argument("prompt", type=str, help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering for sampling")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium", "large"],
                        help="Model size to use")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed(args.seed)
    
    # Print device info
    device_info = get_device_info()
    print(f"\n{'='*50}")
    print(f"xLSTM Inference")
    print(f"{'='*50}")
    print(f"Device: {device_info['device_name']} ({DEVICE})")
    print(f"Model size: {args.model_size}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab_size=256)
    
    # Load or create model
    if args.model_path and Path(args.model_path).exists():
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        config = checkpoint.get('config', get_model_config(args.model_size))
        model = xLSTMLarge(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded from checkpoint")
    else:
        print(f"Creating new {args.model_size} model...")
        config = get_model_config(args.model_size)
        model = xLSTMLarge(config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Move model to device
    model = model.to(DEVICE)
    model.eval()
    
    # Generate text
    print(f"\nGeneration settings:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Max length: {args.max_length}")
    
    start_time = time.time()
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=DEVICE
    )
    
    elapsed_time = time.time() - start_time
    tokens_per_second = args.max_length / elapsed_time
    
    print(f"\nGeneration complete!")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {tokens_per_second:.1f} tokens/second")
    print(f"\nFull generated text:")
    print(f"{'='*50}")
    print(generated_text)
    print(f"{'='*50}")

if __name__ == "__main__":
    main()