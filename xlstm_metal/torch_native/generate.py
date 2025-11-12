#!/usr/bin/env python
"""PyTorch xLSTM Inference Runner (config-driven)"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, List
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from xlstm_metal.torch_native.utils import load_config, resolve_dtype
from xlstm_metal.torch_native.models.wired_xlstm import WiredxLSTM
from xlstm_metal.torch_native.wiring import create_auto_wiring
from xlstm_metal.torch_native.blocks.soft_cap.softcap import soft_cap
from xlstm_metal.torch_native.tokenizer.block import TokenizerBlock, TokenizerConfig


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
        PyTorch:     runner = xLSTMRunner.from_pretrained("NX-AI/xLSTM-7b")
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
        self.output_logit_soft_cap = self.config.get('output_logit_soft_cap', 30.0)
        self.pad_token_id = self.config.get('pad_token_id', 0)
        self.bos_token_id = self.config.get('bos_token_id', 0)
        self.eos_token_id = self.config.get('eos_token_id', None)
        self.force_bos_token_insert = self.config.get('force_bos_token_insert', False)
        self.default_stop_tokens = [self.eos_token_id] if self.eos_token_id is not None else []

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
        self.model.eval()

        print(f"✓ xLSTM NCPS model created with {self.num_blocks} blocks, {self.embedding_dim}d")

        # State for stateful generation
        self.state = None
        # Tokenizer (if present)
        if (self.model_path / "tokenizer.json").exists():
            self.tokenizer = TokenizerBlock(TokenizerConfig(model_path=str(self.model_path)))
        else:
            self.tokenizer = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        model_path = Path(model_id)
        if model_path.exists():
            return cls(model_id, **kwargs)
        try:
            from huggingface_hub import snapshot_download
            downloaded_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
            )
            return cls(downloaded_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load model {model_id}: {e}")

    def reset_state(self):
        """Reset the internal state for stateful generation."""
        self.state = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [B, S]

        Returns:
            logits: Output logits [B, S, vocab_size]
        """
        logits, self.state = self.model(input_ids, self.state, return_last_states=True)
        return soft_cap(logits, self.output_logit_soft_cap)

    @torch.inference_mode()
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
            Next token ID as an integer
        """
        logits = self.forward(input_ids)
        next_logits = logits[0, -1, :]

        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            values, idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            mask = torch.zeros_like(next_logits, dtype=torch.bool)
            mask[idx] = True
            next_logits = torch.where(mask, next_logits, torch.full_like(next_logits, -float('inf')))

        # Top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            keep = cumprobs <= top_p
            # Always keep first token
            keep[0] = True
            cutoff = keep.sum().item()
            mask_idx = sorted_idx[:cutoff]
            mask = torch.zeros_like(next_logits, dtype=torch.bool)
            mask[mask_idx] = True
            next_logits = torch.where(mask, next_logits, torch.full_like(next_logits, -float('inf')))

        probs = torch.softmax(next_logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        return token

    def generate(
            self,
            prompt: str,
            max_tokens: int = 50,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            stop_tokens: Optional[List[int]] = None
    ) -> str:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            stop_tokens: List of token IDs to stop on

        Returns:
            Generated text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available in model directory")
        input_ids = self.tokenizer.encode(prompt).unsqueeze(0)  # [1, S]
        if self.force_bos_token_insert and self.bos_token_id is not None:
            bos = torch.tensor([[self.bos_token_id]], device=input_ids.device)
            input_ids = torch.cat([bos, input_ids], dim=1)
        generated = input_ids.clone()
        stop_tokens = stop_tokens or self.default_stop_tokens
        for _ in range(max_tokens):
            token = self.generate_next_token(generated, temperature=temperature, top_k=top_k, top_p=top_p)
            generated = torch.cat([generated, torch.tensor([[token]], device=generated.device)], dim=1)
            if stop_tokens and token in stop_tokens:
                break
        return self.tokenizer.decode(generated[0])

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
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'force_bos_token_insert': self.force_bos_token_insert,
        }

    def _prepare_prompt(self, prompt_ids) -> List[int]:
        """Normalize prompt IDs and enforce BOS/stop-token defaults."""
        if prompt_ids is None:
            tokens: List[int] = []
        elif hasattr(prompt_ids, "tolist"):
            tokens = prompt_ids.tolist()
        else:
            tokens = list(prompt_ids)

        tokens = [int(t) for t in tokens]

        if not tokens and self.bos_token_id is not None:
            tokens = [int(self.bos_token_id)]
        elif (
            tokens
            and self.force_bos_token_insert
            and self.bos_token_id is not None
            and tokens[0] != int(self.bos_token_id)
        ):
            tokens = [int(self.bos_token_id)] + tokens

        if not tokens:
            raise ValueError(
                "Prompt is empty and no BOS token was available in config.json."
            )

        return tokens


__all__ = ["xLSTMRunner"]
