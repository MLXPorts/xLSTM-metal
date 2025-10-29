#!/usr/bin/env python
"""
Generic xLSTM Inference Runner (Config-Driven)

Provides text generation interface for any xLSTM model size by loading
configuration from the model directory or HuggingFace Hub.
"""

import mlx.core as mx
from typing import Optional, List
from pathlib import Path

from xlstm_metal.blocks.mlx.wiring import WiredMADModel, create_xlstm_wiring
from xlstm_metal.inference.utils import load_config, load_weights_into_wired_model, infer_config_from_checkpoint
from xlstm_metal.blocks.mlx.mlstm import soft_cap


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

    def __init__(self, model_path: str):
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

        # Extract key parameters for easy access
        self.embedding_dim = self.config['embedding_dim']
        self.num_heads = self.config['num_heads']
        self.num_blocks = self.config['num_blocks']
        self.vocab_size = self.config['vocab_size']
        self.output_logit_soft_cap = self.config['output_logit_soft_cap']

        # Create wiring from config
        print(f"Creating xLSTM MAD wiring ({self.num_blocks} blocks, {self.embedding_dim}d)...")
        self.wiring = create_xlstm_wiring(self.config)

        # Create wired model
        print("Creating WiredMADModel...")
        self.model = WiredMADModel(
            wiring=self.wiring,
            input_block='embedding',
            output_block='lm_head'
        )

        print(f"✓ xLSTM model created with {self.num_blocks} blocks, {self.embedding_dim}d")

        # Load weights automatically
        print("Loading weights...")
        self._load_weights()

        # State for stateful generation
        self.state = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False
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
            return cls(model_id)

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
            return cls(downloaded_path)

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

    def _load_weights(self):
        """
        Load pretrained weights from model directory.

        Automatically detects format (safetensors or NPZ) and loads weights.
        """
        from .utils.safetensors_loader import load_safetensors_into_wired_model

        # Check for safetensors files
        if (self.model_path / "model.safetensors").exists() or \
           (self.model_path / "model.safetensors.index.json").exists():
            # Load from safetensors directory
            load_safetensors_into_wired_model(str(self.model_path), self.model)
            print(f"✓ Weights loaded from safetensors")
        elif (self.model_path / "model.npz").exists():
            # Load from NPZ file
            load_weights_into_wired_model(str(self.model_path / "model.npz"), self.model)
            print(f"✓ Weights loaded from NPZ")
        else:
            raise ValueError(
                f"No weights found in {self.model_path}. "
                f"Expected model.safetensors or model.npz"
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
            next_token_logits = next_token_logits / temperature

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
            'execution_stages': len(self.model.stages),
            'block_names': list(self.model.blocks.keys())
        }
