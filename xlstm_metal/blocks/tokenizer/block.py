"""
Tokenizer Block for MAD System

Wraps HuggingFace tokenizers as MAD blocks.
"""

import mlx.core as mx
from typing import Union, List
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer block."""
    model_path: str  # Path to HF model or tokenizer
    vocab_size: int = 50304
    eos_token_id: int = 0
    bos_token_id: int = 0
    pad_token_id: int = 1


class TokenizerBlock:
    """
    Tokenizer block for MAD wiring.

    This wraps a HuggingFace tokenizer and provides encode/decode methods
    that work with MLX arrays.

    Example:
        >>> config = TokenizerConfig(model_path="NX-AI/xLSTM-7b")
        >>> tokenizer = TokenizerBlock(config)
        >>> ids = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(ids)
    """

    def __init__(self, config: TokenizerConfig):
        """
        Initialize tokenizer block.

        Args:
            config: TokenizerConfig with model path and settings
        """
        self.config = config
        self._tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load HuggingFace tokenizer (lazy)."""
        # Don't actually load until first use to avoid import issues
        pass

    def _ensure_tokenizer(self):
        """Ensure tokenizer is loaded."""
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            from pathlib import Path

            # Load tokenizer.json directly (avoids transformers/PIL issues)
            tokenizer_path = Path(self.config.model_path) / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

            # Store special token IDs
            self._eos_token_id = 0  # From tokenizer_config.json
            self._bos_token_id = 0
            self._pad_token_id = 1

    def encode(self, text: Union[str, List[str]]) -> mx.array:
        """
        Encode text to token IDs.

        Args:
            text: String or list of strings to encode

        Returns:
            MLX array of token IDs [S] or [B, S]
        """
        self._ensure_tokenizer()
        if isinstance(text, str):
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
            return mx.array(encoding.ids)
        else:
            # Batch encoding
            encodings = self._tokenizer.encode_batch(text, add_special_tokens=False)
            return mx.array([enc.ids for enc in encodings])

    def decode(self, ids: Union[mx.array, List[int]]) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            ids: Token IDs as MLX array or list

        Returns:
            Decoded text string or list of strings
        """
        self._ensure_tokenizer()
        if isinstance(ids, mx.array):
            ids = ids.tolist()

        # Check if batch (2D list)
        if isinstance(ids[0], list):
            return [self._tokenizer.decode(seq, skip_special_tokens=True) for seq in ids]
        else:
            return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.config.vocab_size

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        self._ensure_tokenizer()
        return self._eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        self._ensure_tokenizer()
        return self._bos_token_id

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        self._ensure_tokenizer()
        return self._pad_token_id
