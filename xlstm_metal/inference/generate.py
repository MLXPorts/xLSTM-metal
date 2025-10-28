#!/usr/bin/env python
"""
Generic xLSTM Inference Runner (Config-Driven)

Provides text generation interface for any xLSTM model size by loading
configuration from the model directory or HuggingFace Hub.
"""

import mlx.core as mx
from typing import Optional, List
from pathlib import Path

from xlstm_metal.mlx_blocks.wiring import create_xlstm_wiring
from xlstm_metal.utils.infer_config_from_checkpoint import infer_config_from_checkpoint
from xlstm_metal.utils.weight_loader import load_weights_into_wired_model
from xlstm_metal.mlx_blocks.mlstm.components import soft_cap


class xLSTMRunner:
    """
    Generic inference runner for xLSTM models (checkpoint-driven).

    Automatically infers model configuration from checkpoint and creates
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
            model_path: Path to model directory containing checkpoint and weights

        The model configuration is inferred automatically from the checkpoint,
        eliminating the need for hardcoded model size parameters.
        """
        self.model_path = Path(model_path)

        # Infer configuration from checkpoint
        print(f"Inferring configuration from checkpoint in {self.model_path}...")
        self.config = infer_config_from_checkpoint(str(self.model_path))

        # Extract key parameters for easy access
        self.embedding_dim = self.config['embedding_dim']
        self.num_heads = self.config['num_heads']
        self.num_blocks = self.config['num_blocks']
        self.vocab_size = self.config['vocab_size']
        self.output_logit_soft_cap = self.config['output_logit_soft_cap']

        # Create wiring from config
        print(f"Creating xLSTM MAD wiring ({self.num_blocks} blocks, {self.embedding_dim}d)...")
        self.wiring = create_xlstm_wiring(self.config)

        print(f"✓ xLSTM neural circuit created with {self.num_blocks} blocks, {self.embedding_dim}d")

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
        from .utils.safetensors_loader import load_safetensors_into_wiring

        # Check for safetensors files
        if (self.model_path / "model.safetensors").exists() or \
           (self.model_path / "model.safetensors.index.json").exists():
            # Load from safetensors directory
            load_safetensors_into_wiring(str(self.model_path), self.wiring)
            print(f"✓ Weights loaded from safetensors")
        elif (self.model_path / "model.npz").exists():
            # Load from NPZ file
            load_weights_into_wired_model(str(self.model_path / "model.npz"), self.wiring)
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
            'total_neurons': len(self.wiring.neurons),
            'execution_stages': len(self.wiring.get_execution_stages()),
            'neuron_names': list(self.wiring.neurons.keys())
        }
