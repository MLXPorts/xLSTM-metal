#!/usr/bin/env python
"""
xLSTM-7B Inference Runner using MAD Wiring

Provides text generation interface for xLSTM-7B model using WiredMADModel.
"""

import mlx.core as mx
from typing import Optional, List
from pathlib import Path

from ..mlx_blocks.wiring import create_xlstm_wiring
from ..utils.weight_loader import load_weights_into_wired_model
from ..mlx_blocks.mlstm.components import soft_cap


class xLSTM7BRunner:
    """
    Inference runner for xLSTM-7B using MAD wiring.

    Example:
        >>> runner = xLSTM7BRunner()
        >>> runner.load_weights("model_cache/xlstm_7b_mlx_converted.npz")
        >>> output = runner.generate("Hello, world!", max_tokens=50)
        >>> print(output)
    """

    def __init__(
        self,
        embedding_dim: int = 4096,
        num_heads: int = 8,
        num_blocks: int = 32,
        vocab_size: int = 50304,
        output_logit_soft_cap: float = 30.0
    ):
        """
        Initialize xLSTM-7B runner.

        Args:
            embedding_dim: Model dimension (default: 4096)
            num_heads: Number of attention heads (default: 8)
            num_blocks: Number of xLSTM blocks (default: 32)
            vocab_size: Vocabulary size (default: 50304)
            output_logit_soft_cap: Output logit soft cap (default: 30.0)
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.output_logit_soft_cap = output_logit_soft_cap

        # Create wiring
        print("Creating xLSTM-7B MAD wiring...")
        config = {
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'num_blocks': num_blocks,
            'vocab_size': vocab_size,
            'qk_dim_factor': 0.5,
            'v_dim_factor': 1.0,
            'ffn_proj_factor': 2.671875,
            'gate_soft_cap': 15.0,
            'norm_eps': 1e-6,
            'output_logit_soft_cap': output_logit_soft_cap
        }
        self.wiring = create_xlstm_wiring(config)

        print(f"✓ xLSTM-7B neural circuit created with {num_blocks} blocks")

        # State for stateful generation
        self.state = None

    def load_weights(self, model_path: str):
        """
        Load pretrained weights from safetensors or NPZ file.

        Args:
            model_path: Path to model directory (for safetensors) or NPZ file
        """
        from pathlib import Path
        from ..utils.safetensors_loader import load_safetensors_into_wiring
        path = Path(model_path)

        if path.is_dir():
            # Load from safetensors directory
            load_safetensors_into_wiring(str(path), self.wiring)
        elif path.suffix == '.npz':
            # Load from NPZ file
            load_weights_into_wired_model(str(path), self.wiring)
        else:
            raise ValueError(f"Unknown model format: {model_path}")

    def reset_state(self):
        """Reset the internal state for stateful generation."""
        self.state = None

    def forward(
        self,
        input_ids: mx.array,
        state: Optional[dict] = None
    ) -> tuple[mx.array, dict]:
        """
        Forward pass through the neural circuit.

        Args:
            input_ids: Input token IDs [B, S]
            state: Optional state dict for stateful generation

        Returns:
            logits: Output logits [B, S, vocab_size]
            new_state: Updated state dict
        """
        # Execute neural circuit
        logits, new_state = self.wiring(input_ids, state)

        # Apply output soft-cap
        logits_capped = soft_cap(logits, self.output_logit_soft_cap)

        return logits_capped, new_state

    def generate_next_token(
        self,
        input_ids: mx.array,
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
        # Forward pass
        logits, self.state = self.forward(input_ids, self.state)

        # Get logits for last token
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            top_k_values, top_k_indices = mx.topk(next_token_logits, top_k)
            # Zero out all non-top-k values
            mask = mx.zeros_like(next_token_logits)
            mask[top_k_indices] = 1
            next_token_logits = mx.where(mask == 1, next_token_logits, -float('inf'))

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits = mx.sort(next_token_logits)[::-1]
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            # Find cutoff index
            cutoff_idx = mx.argmax(cumulative_probs > top_p)
            cutoff_logit = sorted_logits[cutoff_idx]

            # Zero out all values below cutoff
            next_token_logits = mx.where(
                next_token_logits >= cutoff_logit,
                next_token_logits,
                -float('inf')
            )

        # Sample from distribution
        probs = mx.softmax(next_token_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs))

        return int(next_token)

    def generate(
        self,
        prompt_ids: List[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate tokens autoregressively.

        Args:
            prompt_ids: Input prompt token IDs
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            stop_tokens: List of token IDs to stop on

        Returns:
            List of generated token IDs (including prompt)
        """
        # Reset state for new generation
        self.reset_state()

        # Convert prompt to array [1, S]
        generated = list(prompt_ids)
        current_ids = mx.array([prompt_ids])

        # Generate tokens
        for _ in range(max_tokens):
            next_token = self.generate_next_token(
                current_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            generated.append(next_token)

            # Check for stop tokens
            if stop_tokens and next_token in stop_tokens:
                break

            # Update input for next iteration (only use last token for stateful generation)
            current_ids = mx.array([[next_token]])

        return generated

    def get_model_info(self) -> dict:
        """
        Get information about the neural circuit structure.

        Returns:
            Dictionary with model info
        """
        return {
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'vocab_size': self.vocab_size,
            'total_neurons': len(self.wiring.neurons),
            'execution_stages': len(self.wiring.get_execution_stages()),
            'neuron_names': list(self.wiring.neurons.keys())
        }
