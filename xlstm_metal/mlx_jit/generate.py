#!/usr/bin/env python
"""
Generic xLSTM Inference Runner (Config-Driven)

Provides text generation interface for any xLSTM model size by loading
configuration from the model directory or HuggingFace Hub.
"""

# Use importlib to set the correct path for imports
import sys
from pathlib import Path
from typing import Optional, List

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xlstm_metal.mlx_jit.utils import load_config, resolve_dtype
from xlstm_metal.mlx_jit.models.wired_xlstm import WiredxLSTM
from xlstm_metal.mlx_jit.wiring.auto_wiring import create_auto_wiring
from xlstm_metal.mlx_jit.blocks.soft_cap import soft_cap


class xLSTMRunner:
    """
    Generic inference runner for xLSTM models (config-driven).

    Automatically loads model configuration from config.json and creates
    the appropriate model architecture. Works with any xLSTM model size.

    Example:
        >>> # From local directory
        >>> runner = xLSTMRunner("xlstm_7b_model")

        >>> # From HuggingFace Hub
        >>> runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

        >>> # Generate text
        >>> output = runner.generate("Hello, world!", max_tokens=50)
        >>> print(output)

    Comparison to PyTorch:
        PyTorch: model = AutoModelForCausalLM.from_pretrained("NX-AI/xLSTM-7b")
        MLX:     runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")
    """

    def __init__(self, model_path: str, show_wiring: bool = False,
                 wiring_style: str = "unicode"):
        """
        Initialize xLSTM runner from model directory.

        Args:
            model_path: Path to model directory containing config.json and weights

        The model configuration is loaded automatically from config.json,
        eliminating the need for hardcoded model size parameters.
        """
        self.model_path = Path(model_path)

        # Load configuration from model directory
        print(f"Loading configuration from {self.model_path / 'config.json'}...")
        self.config = load_config(str(self.model_path))
        self.compute_dtype = resolve_dtype(self.config.get('autocast_kernel_dtype'))
        self.state_dtype = resolve_dtype(self.config.get('inference_state_dtype'))
        self.norm_reduce_force_float32 = self.config.get('norm_reduction_force_float32', True)

        # Extract key parameters for easy access
        self.embedding_dim = self.config['embedding_dim']
        self.num_heads = self.config['num_heads']
        self.num_blocks = self.config['num_blocks']
        self.vocab_size = self.config['vocab_size']
        self.output_logit_soft_cap = self.config['output_logit_soft_cap']

        # Build NCPS wiring/model
        print(f"Creating NCPS auto-wiring ({self.num_blocks} blocks, {self.embedding_dim}d)...")
        self.wiring = create_auto_wiring(str(self.model_path), self.config)
        if show_wiring:
            self.wiring.print_diagram(include_sensory=False, style=wiring_style)

        print("Creating WiredxLSTM model...")
        self.model = WiredxLSTM(
            wiring=self.wiring,
            load_weights=True,
            model_dir=self.model_path,
            compute_dtype=self.compute_dtype,
            state_dtype=self.state_dtype,
            norm_reduce_force_float32=self.norm_reduce_force_float32,
        )

        print(f"✓ xLSTM NCPS model created with {self.num_blocks} blocks, {self.embedding_dim}d")

        # State for stateful generation
        self.state = None

    @classmethod
    def from_pretrained(
            cls,
            model_id: str,
            cache_dir: Optional[str] = None,
            force_download: bool = False,
            show_wiring: bool = False,
            wiring_style: str = "unicode"
    ):
        """
        Load xLSTM model from HuggingFace Hub or local directory.

        This method matches the PyTorch transformers API and automatically
        downloads the model if needed.

        Args:
            model_id: HuggingFace model ID (e.g., "NX-AI/xLSTM-7b") or local path
            cache_dir: Directory to cache downloaded models (default: ~/.cache/huggingface)
            force_download: Force re-download even if cached

        Returns:
            xLSTMRunner instance with loaded model

        Example:
            >>> # Download from HuggingFace Hub
            >>> runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")

            >>> # Use local directory
            >>> runner = xLSTMRunner.from_pretrained("./local_model")

            >>> # Custom cache directory
            >>> runner = xLSTMRunner.from_pretrained(
            ...     "NX-AI/xLSTM-7b",
            ...     cache_dir="./my_cache"
            ... )
        """
        model_path = Path(model_id)

        # Check if it's a local path
        if model_path.exists():
            print(f"Loading from local directory: {model_id}")
            return cls(model_id, show_wiring=show_wiring, wiring_style=wiring_style)

        # Try to download from HuggingFace Hub
        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading model from HuggingFace Hub: {model_id}")
            downloaded_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
            )
            print(f"✓ Model downloaded to: {downloaded_path}")
            return cls(downloaded_path, show_wiring=show_wiring, wiring_style=wiring_style)

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models from HuggingFace. "
                "Install it with: pip install huggingface_hub"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load model '{model_id}'. "
                f"It's not a local directory and failed to download from HuggingFace Hub. "
                f"Error: {e}"
            )

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
        logits, new_state = self.model(
            input_ids,
            state,
            return_last_states=True
        )

        # Apply output soft-cap
        logits_capped = soft_cap(logits, self.output_logit_soft_cap)

        return logits_capped, new_state

    def generate_next_token(
            self,
            input_ids: mx.array,
            temperature: mx.array = mx.array(1.0 ,dtype= mx.float32),
            top_k: Optional[mx.array] = None,
            top_p: Optional[mx.array] = None
    ) -> mx.array:
        """
        Generate next token given input token IDs.

        Args:
            input_ids: Input token IDs [1, S]
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
            top_p: Nucleus sampling threshold (None for no filtering)

        Returns:
            Next token ID as an MLX scalar array
        """
        # Forward pass
        logits, self.state = self.forward(input_ids, self.state)

        # Get logits for last token
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / mx.array(temperature, dtype=next_token_logits.dtype)

        neg_inf = mx.full(next_token_logits.shape, -mx.inf, dtype=next_token_logits.dtype)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, next_token_logits.shape[-1])
            topk_values = mx.topk(next_token_logits, k=k)
            kth_value = mx.min(topk_values)
            mask = next_token_logits >= kth_value
            next_token_logits = mx.where(mask, next_token_logits, neg_inf)

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_indices = mx.argsort(next_token_logits)[::-1]
            sorted_logits = mx.take(next_token_logits, sorted_indices)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            top_p_tensor = mx.array(top_p, dtype=sorted_probs.dtype)
            keep_mask_sorted = cumulative_probs <= top_p_tensor

            positions = mx.arange(keep_mask_sorted.shape[0])
            first_position = positions == 0
            keep_mask_sorted = mx.where(
                first_position,
                mx.ones_like(keep_mask_sorted),
                keep_mask_sorted
            )

            keep_count = mx.sum(keep_mask_sorted)
            keep_count = mx.maximum(keep_count, mx.ones_like(keep_count))
            last_index = keep_count - mx.ones_like(keep_count)

            threshold = mx.take(sorted_logits, last_index)
            mask = next_token_logits >= threshold
            next_token_logits = mx.where(mask, next_token_logits, neg_inf)

        # Sample from distribution
        probs = mx.softmax(next_token_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs))

        return next_token

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
        prompt_array = mx.array(prompt_ids, mx.int64)
        generated = prompt_array
        current_ids = mx.expand_dims(prompt_array, axis=0)
        stop_tokens_arr = mx.array(stop_tokens) if stop_tokens else None

        # Generate tokens
        for _ in range(max_tokens):
            next_token = self.generate_next_token(
                current_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            next_token_vector = mx.reshape(next_token, (1,))
            generated = mx.concatenate([generated, next_token_vector])

            # Check for stop tokens
            if stop_tokens_arr is not None and mx.any(next_token == stop_tokens_arr).tolist():
                break

            # Update input for next iteration (only use last token for stateful generation)
            current_ids = mx.reshape(next_token, (1, 1))

        return generated.tolist()

    def get_model_info(self) -> dict:
        """
        Get information about the model structure.

        Returns:
            Dictionary with model info including config values
        """
        return {
            'model_path': str(self.model_path),
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'vocab_size': self.vocab_size,
            'qk_dim': self.config['qk_dim'],
            'v_dim': self.config['v_dim'],
            'chunk_size': self.config.get('chunk_size', 64),
            'gate_soft_cap': self.config['gate_soft_cap'],
            'output_logit_soft_cap': self.output_logit_soft_cap,
            'total_blocks': len(self.model.blocks),
            'block_types': [
                self.wiring.get_block_info(i)['type']
                for i in range(self.num_blocks)
            ],
            'has_embedding': self.wiring.structure['has_embedding'],
            'has_out_norm': self.wiring.structure['has_out_norm'],
            'has_lm_head': self.wiring.structure['has_lm_head'],
        }
