#!/usr/bin/env python
"""
Tokenizer Block for MAD System

Wraps HuggingFace tokenizers as MAD blocks.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import Tokenizer as _HFTokenizer
except ImportError:
    _HFTokenizer = None

import mlx.core as mx


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer block."""
    model_path: str  # Path to HF model or tokenizer
    vocab_size: int = field(default_factory=lambda: 50304)
    eos_token_id: int = field(default_factory=lambda: 0)
    bos_token_id: int = field(default_factory=lambda: 0)
    pad_token_id: int = field(default_factory=lambda: 1)


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
        self._special_tokens = None
        self._vocab_size = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load HuggingFace tokenizer (lazy)."""
        # Don't actually load until first use to avoid import issues
        pass

    def _ensure_tokenizer(self):
        """Ensure tokenizer is loaded."""
        if self._tokenizer is None:
            if _HFTokenizer is None:
                raise ImportError(
                    "The 'tokenizers' package is required. Install it via pip install tokenizers."
                )
            # Load tokenizer.json directly (avoids transformers/PIL issues)
            tokenizer_path = Path(self.config.model_path) / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            self._tokenizer = _HFTokenizer.from_file(str(tokenizer_path))

            # Load tokenizer_config.json for accurate metadata
            config_path = tokenizer_path.with_name("tokenizer_config.json")
            if config_path.exists():
                with open(config_path) as cfg_file:
                    tokenizer_cfg = json.load(cfg_file)
            else:
                tokenizer_cfg = {}

            self._special_tokens = {
                "eos": tokenizer_cfg.get("eos_token_id", self.config.eos_token_id),
                "bos": tokenizer_cfg.get("bos_token_id", self.config.bos_token_id),
                "pad": tokenizer_cfg.get("pad_token_id", self.config.pad_token_id),
            }
            self._vocab_size = tokenizer_cfg.get(
                "vocab_size", self._tokenizer.get_vocab_size()
            )

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
        self._ensure_tokenizer()
        return self._vocab_size or self.config.vocab_size

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        self._ensure_tokenizer()
        return self._special_tokens["eos"]

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        self._ensure_tokenizer()
        return self._special_tokens["bos"]

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        self._ensure_tokenizer()
        return self._special_tokens["pad"]
