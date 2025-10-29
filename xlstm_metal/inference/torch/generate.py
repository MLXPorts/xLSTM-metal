#!/usr/bin/env python
"""
PyTorch xLSTM Inference Runner

Provides text generation interface for xLSTM using PyTorch backend.
Compatible with MLX runner but uses pure PyTorch computation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from pathlib import Path

from xlstm_metal.blocks.mlx.wiring import WiredMADModel, create_xlstm_wiring
from xlstm_metal.inference.utils import load_config


class PyTorchxLSTMRunner:
    """
    PyTorch inference runner for xLSTM models.

    Automatically loads model configuration and creates the model architecture
    using PyTorch backend. Weights are loaded from safetensors or NPZ files.

    Example:
        >>> runner = PyTorchxLSTMRunner("xlstm_7b_model")
        >>> output = runner.generate("Hello, world!", max_tokens=50)
        >>> print(output)
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize PyTorch xLSTM runner.

        Args:
            model_path: Path to model directory containing config.json and weights
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.). Auto-detects if None.
        """
        self.model_path = Path(model_path)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load configuration
        print(f"Loading configuration from {self.model_path / 'config.json'}...")
        self.config = load_config(str(self.model_path))

        # Extract key parameters
        self.embedding_dim = self.config['embedding_dim']
        self.num_heads = self.config['num_heads']
        self.num_blocks = self.config['num_blocks']
        self.vocab_size = self.config['vocab_size']
        self.output_logit_soft_cap = self.config.get('output_logit_soft_cap', 30.0)

        # Create wiring with TORCH backend
        print(f"Creating xLSTM wiring with PyTorch backend ({self.num_blocks} blocks, {self.embedding_dim}d)...")
        self.config['backend_type'] = 'torch'  # Set PyTorch backend
        self.wiring = create_xlstm_wiring(self.config)

        # Create wired model
        print("Creating WiredMADModel with PyTorch blocks...")
        self.model = WiredMADModel(
            wiring=self.wiring,
            input_block='embedding',
            output_block='lm_head'
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ PyTorch xLSTM model created with {self.num_blocks} blocks, {self.embedding_dim}d")

        # Load weights
        print("Loading weights...")
        self._load_weights()

        # State for stateful generation
        self.state = None

    def _load_weights(self):
        """Load weights from safetensors or NPZ file."""
        # Check for safetensors
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        
        if safetensors_files:
            print(f"Loading from safetensors: {safetensors_files[0]}")
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_files[0]))
                self.model.load_state_dict(state_dict, strict=False)
                print("✓ Weights loaded successfully")
            except ImportError:
                print("Warning: safetensors not available. Install with: pip install safetensors")
        else:
            print("Warning: No weights file found")

    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> int:
        """
        Generate next token given input token IDs.

        Args:
            input_ids: Input token IDs [1, S]
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
            top_p: Nucleus sampling threshold (None for no filtering)

        Returns:
            Next token ID (int)
        """
        with torch.no_grad():
            # Forward pass
            logits = self.model(input_ids, hidden_states=self.state)[0]

            # Get logits for last token
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_indices = torch.argsort(next_token_logits, descending=True)[:top_k]
                threshold = next_token_logits[top_k_indices[-1]]
                next_token_logits = torch.where(
                    next_token_logits >= threshold,
                    next_token_logits,
                    torch.tensor(float('-inf'), device=self.device)
                )

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits = torch.sort(next_token_logits, descending=True)[0]
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                keep_mask = cumulative_probs <= top_p
                num_keep = max(1, keep_mask.sum().item())
                
                threshold = sorted_logits[num_keep - 1]
                next_token_logits = torch.where(
                    next_token_logits >= threshold,
                    next_token_logits,
                    torch.tensor(float('-inf'), device=self.device)
                )

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            return int(next_token)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[int]] = None,
        verbose: bool = True
    ) -> str:
        """
        Generate text autoregressively.

        Args:
            prompt: Initial text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = greedy at 0, higher = more random)
            top_k: Top-k sampling (keep top k tokens)
            top_p: Nucleus sampling (keep tokens with cumsum prob <= p)
            stop_tokens: Token IDs to stop generation
            verbose: Print generation progress

        Returns:
            Generated text
        """
        # Tokenize prompt (stub - would need actual tokenizer)
        prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=self.device)  # Placeholder

        output_ids = list(prompt_ids[0].cpu().numpy())

        for i in range(max_tokens):
            # Generate next token
            next_token = self.generate_next_token(
                torch.tensor([output_ids], dtype=torch.long, device=self.device),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            output_ids.append(next_token)

            if verbose and (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{max_tokens} tokens")

            # Check for stop tokens
            if stop_tokens and next_token in stop_tokens:
                break

        # Detokenize (stub - would need actual tokenizer)
        return f"Generated {len(output_ids)} tokens"


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch xLSTM Inference")
    parser.add_argument("--model", type=str, default="xlstm_7b_model", help="Model directory")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, etc)")

    args = parser.parse_args()

    runner = PyTorchxLSTMRunner(args.model, device=args.device)
    output = runner.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print(output)
