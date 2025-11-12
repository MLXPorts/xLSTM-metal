#!/usr/bin/env python
"""Tokenizer Block for MAD System (PyTorch backend)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Optional

try:
    from tokenizers import Tokenizer as _HFTokenizer
except ImportError:
    _HFTokenizer = None

import torch


@dataclass
class TokenizerConfig:
    model_path: str
    vocab_size: int = field(default_factory=lambda: 50304)
    eos_token_id: int = field(default_factory=lambda: 0)
    bos_token_id: int = field(default_factory=lambda: 0)
    pad_token_id: int = field(default_factory=lambda: 1)


class TokenizerBlock:
    def __init__(self, config: TokenizerConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
        self._tokenizer = None
        self._special_tokens = None
        self._vocab_size = None
        self._load_tokenizer_lazy()

    def _load_tokenizer_lazy(self):
        pass

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            if _HFTokenizer is None:
                raise ImportError("The 'tokenizers' package is required. Install it via pip install tokenizers.")
            tokenizer_path = Path(self.config.model_path) / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            self._tokenizer = _HFTokenizer.from_file(str(tokenizer_path))
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
            self._vocab_size = tokenizer_cfg.get("vocab_size", self._tokenizer.get_vocab_size())

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        self._ensure_tokenizer()
        if isinstance(text, str):
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
            return torch.tensor(encoding.ids, dtype=torch.long, device=self.device)
        else:
            encodings = self._tokenizer.encode_batch(text, add_special_tokens=False)
            # Ragged batch -> pad to max length
            max_len = max(len(enc.ids) for enc in encodings) if encodings else 0
            batch_ids = []
            for enc in encodings:
                ids = enc.ids
                if len(ids) < max_len:
                    ids = ids + [self.pad_token_id] * (max_len - len(ids))
                batch_ids.append(ids)
            return torch.tensor(batch_ids, dtype=torch.long, device=self.device)

    def decode(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        self._ensure_tokenizer()
        if isinstance(ids, torch.Tensor):
            ids_list = ids.tolist()
        else:
            ids_list = ids
        if isinstance(ids_list[0], list):
            return [self._tokenizer.decode(seq, skip_special_tokens=True) for seq in ids_list]
        else:
            return self._tokenizer.decode(ids_list, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        self._ensure_tokenizer()
        return self._vocab_size or self.config.vocab_size

    @property
    def eos_token_id(self) -> int:
        self._ensure_tokenizer()
        return self._special_tokens["eos"]

    @property
    def bos_token_id(self) -> int:
        self._ensure_tokenizer()
        return self._special_tokens["bos"]

    @property
    def pad_token_id(self) -> int:
        self._ensure_tokenizer()
        return self._special_tokens["pad"]

__all__ = ["TokenizerBlock", "TokenizerConfig"]
