#!/usr/bin/env python
"""
xLSTM-7B Text Generator with Tokenizer Block

End-to-end text generation using MAD wiring with tokenizer integration.
"""

import mlx.core as mx
from typing import Optional
from pathlib import Path

from ..wiring.core import MADWiring, BlockSpec, BlockType, BackendType
from ..wiring.mlx import create_xlstm_7b_wiring, WiredMADModel
from ..utils.weight_loader import load_weights_into_wired_model
from ..blocks.mlstm_mlx.components import soft_cap


class xLSTM7BTextGenerator:
    """
    Complete text generation system with tokenizer as a MAD block.

    Example:
        >>> generator = xLSTM7BTextGenerator()
        >>> generator.load_weights("model_cache/xlstm_7b_mlx_converted.npz")
        >>> output = generator.generate("Hello, world!", max_tokens=50)
        >>> print(output)
    """

    def __init__(
        self,
        model_path: str = "model_cache/models--NX-AI--xLSTM-7b/snapshots/9dc507bd0939cf372a4a4f667335651d8e49dddb",
        embedding_dim: int = 4096,
        num_heads: int = 8,
        num_blocks: int = 32,
        vocab_size: int = 50304,
        output_logit_soft_cap: float = 30.0,
        debug: bool = False
    ):
        """
        Initialize text generator with tokenizer.

        Args:
            model_path: Path to HF tokenizer
            embedding_dim: Model dimension (default: 4096)
            num_heads: Number of attention heads (default: 8)
            num_blocks: Number of xLSTM blocks (default: 32)
            vocab_size: Vocabulary size (default: 50304)
            output_logit_soft_cap: Output logit soft cap (default: 30.0)
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.output_logit_soft_cap = output_logit_soft_cap
        self.debug = debug

        # Create wiring with model blocks
        print("Creating xLSTM-7B wiring...")
        self.wiring = create_xlstm_7b_wiring(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            output_logit_soft_cap=output_logit_soft_cap
        )

        # Create model
        print("Creating WiredMADModel...")
        self.model = WiredMADModel(self.wiring, 'embedding', 'lm_head', debug=debug)

        # Create tokenizer separately (NOT part of execution graph)
        print(f"Creating tokenizer (model_path={model_path})...")
        from ..blocks.tokenizer.block import TokenizerBlock, TokenizerConfig
        tokenizer_config = TokenizerConfig(model_path=model_path, vocab_size=vocab_size)
        self.tokenizer = TokenizerBlock(tokenizer_config)

        print(f"✓ Text generator initialized")

    def load_weights(self, npz_path: str):
        """
        Load pretrained weights from NPZ file.

        Args:
            npz_path: Path to xlstm_7b_mlx_converted.npz
        """
        load_weights_into_wired_model(npz_path, self.model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            stream: If True, print tokens as generated

        Returns:
            Generated text (prompt + completion)
        """
        # Encode prompt using tokenizer block
        prompt_ids = self.tokenizer.encode(prompt)

        # Prepend BOS token (CRITICAL - from HF README)
        bos_id = self.tokenizer.bos_token_id
        bos_tensor = mx.array([bos_id])
        prompt_ids_with_bos = mx.concatenate([bos_tensor, prompt_ids])

        # Add batch dimension
        if prompt_ids_with_bos.ndim == 1:
            prompt_ids_with_bos = mx.expand_dims(prompt_ids_with_bos, 0)

        if stream:
            print(prompt, end='', flush=True)

        # Generate tokens
        state = None
        current_ids = prompt_ids_with_bos
        generated_tokens = []

        for _ in range(max_tokens):
            # Forward pass
            logits, state = self.model(current_ids, state)

            # Get last token logits and apply soft cap
            next_token_logits = logits[0, -1, :]
            next_token_logits = soft_cap(next_token_logits, self.output_logit_soft_cap)

            # Handle temperature=0 as greedy decoding
            if temperature == 0.0:
                next_token_id = int(mx.argmax(next_token_logits))
            else:
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k
                if top_k is not None:
                    top_k_values, top_k_indices = mx.topk(next_token_logits, top_k)
                    mask = mx.zeros_like(next_token_logits)
                    mask[top_k_indices] = 1
                    next_token_logits = mx.where(mask == 1, next_token_logits, -float('inf'))

                # Apply top-p
                if top_p is not None:
                    sorted_logits = mx.sort(next_token_logits)[::-1]
                    sorted_probs = mx.softmax(sorted_logits, axis=-1)
                    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                    cutoff_idx = mx.argmax(cumulative_probs > top_p)
                    cutoff_logit = sorted_logits[cutoff_idx]
                    next_token_logits = mx.where(
                        next_token_logits >= cutoff_logit,
                        next_token_logits,
                        -float('inf')
                    )

                # Sample
                probs = mx.softmax(next_token_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs))
                next_token_id = int(next_token)

            # Check for EOS (but skip if EOS == BOS, since BOS is prepended)
            if (next_token_id == self.tokenizer.eos_token_id and
                self.tokenizer.eos_token_id != self.tokenizer.bos_token_id):
                break

            # Track generated token
            generated_tokens.append(next_token_id)

            # Decode and print if streaming
            if stream:
                token_text = self.tokenizer.decode([next_token_id])
                print(token_text, end='', flush=True)

            # Update for next iteration (stateful - only use new token)
            current_ids = mx.array([[next_token_id]])

        if stream:
            print()  # New line

        # Decode full sequence: prompt (with BOS) + generated tokens
        all_ids = prompt_ids_with_bos[0].tolist() + generated_tokens
        return self.tokenizer.decode(all_ids)
