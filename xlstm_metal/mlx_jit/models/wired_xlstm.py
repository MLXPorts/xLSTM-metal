"""Wired xLSTM Model - NCPS-style model-agnostic wrapper.

This module follows NCPS patterns to create a fully wired xLSTM model
from safetensors structure. It automatically detects block types (mLSTM,
sLSTM, conv1d attention, etc.) and builds the appropriate architecture.

Architecture:
    embedding -> blocks (mLSTM/sLSTM/attention) -> out_norm -> lm_head

The wiring system determines:
- Number of blocks
- Type of each block
- Connections between blocks
- Model structure

Example:
    >>> from xlstm_metal.mlx_jit.wiring import create_auto_wiring
    >>> from xlstm_metal.mlx_jit.models import WiredxLSTM
    >>>
    >>> # Auto-detect model structure
    >>> wiring = create_auto_wiring("xlstm_7b_model")
    >>> model = WiredxLSTM(wiring=wiring)

    >>> # Or use convenience method
    >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.wiring import AutoWiring, create_auto_wiring
from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards


class WiredxLSTM(nn.Module):
    """
    Model-agnostic xLSTM model using NCPS-style wiring.

    Automatically builds the correct architecture based on safetensors structure.
    Supports:
    - mLSTM blocks (matrix memory)
    - sLSTM blocks (scalar memory)
    - Conv1d attention blocks
    - Mixed architectures

    Args:
        wiring: AutoWiring object that defines model structure
        load_weights: Whether to load pretrained weights from safetensors
        model_dir: Optional model directory (for weight loading)

    Attributes:
        wiring: The wiring object defining architecture
        embedding: Token embedding layer
        blocks: List of xLSTM blocks (mLSTM/sLSTM/attention)
        out_norm: Optional output normalization
        lm_head: Language model head for logits
    """

    def __init__(
            self,
            wiring: AutoWiring,
            load_weights: bool = False,
            model_dir: Optional[Union[str, Path]] = None,
            compute_dtype: mx.Dtype = mx.float32,
            state_dtype: mx.Dtype = mx.float32,
            norm_reduce_force_float32: bool = True,
    ):
        super().__init__()

        self.wiring = wiring
        self.config = wiring.config
        self.model_dir = Path(model_dir) if model_dir else wiring.model_dir
        self.compute_dtype = compute_dtype
        self.state_dtype = state_dtype
        self.norm_reduce_force_float32 = norm_reduce_force_float32

        # Token / embedding metadata from config.json
        self.pad_token_id = self.config.get('pad_token_id')
        self.bos_token_id = self.config.get('bos_token_id')
        self.eos_token_id = self.config.get('eos_token_id')
        self.force_bos_token_insert = self.config.get('force_bos_token_insert', False)
        self.tie_word_embeddings = self.config.get('tie_word_embeddings', False)
        self.add_embedding_dropout = self.config.get('add_embedding_dropout', False)
        self.embedding_dropout_prob = float(self.config.get('embedding_dropout', 0.0) or 0.0)
        self.embedding_dropout: Optional[nn.Module] = None
        self._embeddings_tied = False

        # Build architecture from wiring
        self._build_model()

        # Optionally load weights
        if load_weights:
            self.load_pretrained_weights()

        # Default to eval() for inference configs to disable dropout noise
        if self.config.get('mode', 'inference') == 'inference':
            self.eval()

    def _build_model(self):
        """Build model architecture from wiring specification."""
        # Embedding layer
        if self.wiring.structure['has_embedding']:
            self.embedding = nn.Embedding(
                num_embeddings=self.config['vocab_size'],
                dims=self.config['embedding_dim']
            )
            self.embedding.weight = mx.array(self.embedding.weight, dtype=self.compute_dtype)

            if self.add_embedding_dropout and self.embedding_dropout_prob > 0.0:
                self.embedding_dropout = nn.Dropout(self.embedding_dropout_prob)
            else:
                self.embedding_dropout = None
        else:
            self.embedding = None
            self.embedding_dropout = None

        # Build blocks based on wiring
        self.blocks = []
        num_blocks = self.wiring.structure['num_blocks']

        for block_idx in range(num_blocks):
            block_info = self.wiring.get_block_info(block_idx)
            block_type = block_info['type']

            # Create appropriate cell based on detected type
            if block_type == 'mlstm':
                cell = self.wiring.create_block_cell(
                    block_idx,
                    compute_dtype=self.compute_dtype,
                    state_dtype=self.state_dtype,
                    norm_reduction_force_float32=self.norm_reduce_force_float32,
                )
                self.blocks.append(cell)
            elif block_type == 'slstm':
                # TODO: Implement sLSTM cell creation
                raise NotImplementedError(f"sLSTM blocks not yet implemented (block {block_idx})")
            elif block_type == 'attention':
                # TODO: Implement attention cell creation
                raise NotImplementedError(f"Attention blocks not yet implemented (block {block_idx})")
            else:
                raise ValueError(f"Unknown block type: {block_type} (block {block_idx})")

        # Output normalization
        if self.wiring.structure['has_out_norm']:
            self.out_norm = nn.RMSNorm(
                dims=self.config['embedding_dim'],
                eps=self.config.get('norm_eps', 1e-6)
            )
        else:
            self.out_norm = None

        # Language model head
        if self.wiring.structure['has_lm_head']:
            self.lm_head = nn.Linear(
                input_dims=self.config['embedding_dim'],
                output_dims=self.config['vocab_size'],
                bias=False
            )
            self.lm_head.weight = mx.array(self.lm_head.weight, dtype=self.compute_dtype)
        else:
            self.lm_head = None

        self._apply_weight_tying()

    def __call__(
            self,
            input_ids: mx.array,
            state: Optional[List[Tuple]] = None,
            return_last_states: bool = False
    ) -> Union[mx.array, Tuple[mx.array, List[Tuple]]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [B, S]
            state: Optional list of states for each block
            return_last_states: Whether to return final states

        Returns:
            logits: Output logits [B, S, vocab_size]
            states: (optional) List of final states for each block
        """
        # Embedding
        if self.embedding is not None:
            x = self.embedding(input_ids)  # [B, S, D]
            if self.embedding_dropout is not None:
                x = self.embedding_dropout(x)
        else:
            x = input_ids

        # Initialize states if not provided
        if state is None:
            state = [None] * len(self.blocks)

        # Process through blocks
        new_states = []
        for block_idx, block in enumerate(self.blocks):
            # print(f"Processing block {block_idx}...")
            # print(f"x shape: {x.shape}")
            # print(f"state shape: {state[block_idx]}")
            # print(f"block: {block}")
            x, block_state = block(x, state[block_idx])
            new_states.append(block_state)

        # Output normalization
        if self.out_norm is not None:
            x = self.out_norm(x)

        # Language model head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x

        if return_last_states:
            return logits, new_states
        return logits

    def load_pretrained_weights(self):
        """Load pretrained weights from safetensors files."""
        if self.model_dir is None:
            raise ValueError("model_dir must be provided to load weights")

        # Use shared loader to read every shard via mx.load
        weights = load_safetensor_shards(str(self.model_dir))
        self._load_weights_from_dict(weights)

    def _load_weights_from_dict(self, weights_dict: Dict[str, mx.array]):
        """Map safetensors weights to model parameters."""
        def _to_compute(tensor: mx.array) -> mx.array:
            return mx.array(tensor, dtype=self.compute_dtype)

        # Load embedding
        if self.embedding is not None and 'backbone.embeddings.weight' in weights_dict:
            self.embedding.weight = _to_compute(weights_dict['backbone.embeddings.weight'])

        # Load blocks
        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, 'get_weight_keys'):
                weight_mapping = block.get_weight_keys()
                for param_path, safetensors_key in weight_mapping.items():
                    if safetensors_key in weights_dict:
                        # Navigate to parameter using path
                        parts = param_path.split('.')
                        obj = block
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], _to_compute(weights_dict[safetensors_key]))

        # Load output norm
        if self.out_norm is not None and 'backbone.out_norm.weight' in weights_dict:
            self.out_norm.weight = _to_compute(weights_dict['backbone.out_norm.weight'])

        # Load lm_head
        if self.lm_head is not None and 'lm_head.weight' in weights_dict:
            self.lm_head.weight = _to_compute(weights_dict['lm_head.weight'])

        self._apply_weight_tying()

    @classmethod
    def from_pretrained(
            cls,
            model_dir: Union[str, Path],
            load_weights: bool = True,
            **kwargs
    ) -> "WiredxLSTM":
        """
        Create model from pretrained checkpoint directory.

        Automatically detects model structure from safetensors and builds
        the appropriate architecture.

        Args:
            model_dir: Path to model directory with safetensors and config
            load_weights: Whether to load pretrained weights
            **kwargs: Additional arguments for model initialization

        Returns:
            Initialized WiredxLSTM model

        Example:
            >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
            >>> logits = model(input_ids)
        """
        model_dir = Path(model_dir)

        # Create auto-wiring from model structure
        wiring = create_auto_wiring(str(model_dir))

        # Build model
        return cls(
            wiring=wiring,
            load_weights=load_weights,
            model_dir=model_dir,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'model_type': 'wired_xlstm',
            'wiring_config': self.wiring.get_config() if hasattr(self.wiring, 'get_config') else None,
            'num_blocks': self.wiring.structure['num_blocks'],
            'block_types': {idx: self.wiring.get_block_info(idx)['type']
                            for idx in range(self.wiring.structure['num_blocks'])},
            'embedding_dim': self.config['embedding_dim'],
            'vocab_size': self.config['vocab_size'],
            'has_embedding': self.wiring.structure['has_embedding'],
            'has_out_norm': self.wiring.structure['has_out_norm'],
            'has_lm_head': self.wiring.structure['has_lm_head'],
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'tie_word_embeddings': self.tie_word_embeddings,
            'add_embedding_dropout': self.add_embedding_dropout,
        }

    def _apply_weight_tying(self) -> None:
        """Tie LM head weights to embeddings when the config requests it."""
        should_tie = bool(self.tie_word_embeddings) and self.embedding is not None and self.lm_head is not None
        self._embeddings_tied = should_tie
        if should_tie:
            self.lm_head.weight = self.embedding.weight


__all__ = ['WiredxLSTM']
