#!/usr/bin/env python
"""
xLSTM-7B Inference Runner using MAD Wiring

Provides text generation interface for xLSTM-7B model using WiredMADModel.
"""

from typing import Optional, List

import mlx.core as mx

from xlstm_metal.blocks.mlstm import soft_cap
from xlstm_metal.mlx_jit.wiring import WiredMADModel, create_xlstm_7b_wiring
from xlstm_metal.mlx_jit.utils import load_weights_into_wired_model


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
        self.wiring = create_xlstm_7b_wiring(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            output_logit_soft_cap=output_logit_soft_cap
        )

        # Create wired model
        print("Creating WiredMADModel...")
        self.model = WiredMADModel(
            wiring=self.wiring,
            input_block='embedding',
            output_block='lm_head'
        )

        print(f"✓ xLSTM-7B model created with {num_blocks} blocks")

        # State for stateful generation
        self.state = None

    def load_weights(self, model_path: str):
        """
        Load pretrained weights from safetensors or NPZ file.

        Args:
            model_path: Path to model directory (for safetensors) or NPZ file
        """
        from pathlib import Path
        from ..utils.safetensors_loader import load_safetensors_into_wired_model
        path = Path(model_path)

        if path.is_dir():
            # Load from safetensors directory
            load_safetensors_into_wired_model(str(path), self.model)
        elif path.suffix == '.npz':
            # Load from NPZ file
            load_weights_into_wired_model(str(path), self.model)
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
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [B, S]
            state: Optional state dict for stateful generation

        Returns:
            logits: Output logits [B, S, vocab_size]
            new_state: Updated state dict
        """
        # WiredMADModel forward returns (output, hidden_states)
        logits, new_state = self.model(input_ids, state)

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
            next_token_logits /= temperature

        # Apply top-k filtering
        if top_k is not None:
            # Get top-k values using argsort (which returns indices we can use)
            top_k_indices = mx.argsort(next_token_logits)[::-1][:top_k]
            # Get threshold value (smallest value in top-k)
            threshold = next_token_logits[top_k_indices[-1]]
            # Keep only top-k or better
            next_token_logits = mx.where(
                next_token_logits >= threshold,
                next_token_logits,
                mx.array(-float('inf'))
            )

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            # Sort logits in descending order
            sorted_logits = mx.sort(next_token_logits)[::-1]
            # Compute probabilities and cumulative sum
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
            
            # Find how many tokens to keep (at least 1)
            keep_mask = cumulative_probs <= top_p
            num_keep = max(1, int(mx.sum(keep_mask).item()))
            
            # Get threshold (logit value at cutoff)
            threshold = sorted_logits[num_keep - 1]
            
            # Keep only tokens above threshold
            next_token_logits = mx.where(
                next_token_logits >= threshold,
                next_token_logits,
                mx.array(-float('inf'))
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
        Get information about the model structure.

        Returns:
            Dictionary with model info
        """
        return {
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'vocab_size': self.vocab_size,
            'total_blocks': len(self.model.blocks),
            'execution_stages': len(self.model.stages),
            'block_names': list(self.model.blocks.keys())
        }
