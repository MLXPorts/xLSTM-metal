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
    >>>
    >>> # Or use convenience method
    >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.wiring import AutoWiring, create_auto_wiring


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
    ):
        super().__init__()

        self.wiring = wiring
        self.config = wiring.config
        self.model_dir = Path(model_dir) if model_dir else wiring.model_dir

        # Build architecture from wiring
        self._build_model()

        # Optionally load weights
        if load_weights:
            self.load_pretrained_weights()

    def _build_model(self):
        """Build model architecture from wiring specification."""
        # Embedding layer
        if self.wiring.structure['has_embedding']:
            self.embedding = nn.Embedding(
                num_embeddings=self.config['vocab_size'],
                dims=self.config['embedding_dim']
            )
        else:
            self.embedding = None

        # Build blocks based on wiring
        self.blocks = []
        num_blocks = self.wiring.structure['num_blocks']

        for block_idx in range(num_blocks):
            block_info = self.wiring.get_block_info(block_idx)
            block_type = block_info['type']

            # Create appropriate cell based on detected type
            if block_type == 'mlstm':
                cell = self.wiring.create_block_cell(block_idx)
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
        else:
            self.lm_head = None

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
        else:
            x = input_ids

        # Initialize states if not provided
        if state is None:
            state = [None] * len(self.blocks)

        # Process through blocks
        new_states = []
        for block_idx, block in enumerate(self.blocks):
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

        # Load index
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Safetensors index not found: {index_path}")

        import json
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by shard
        shard_weights = {}
        for weight_key, shard_file in index['weight_map'].items():
            if shard_file not in shard_weights:
                shard_weights[shard_file] = []
            shard_weights[shard_file].append(weight_key)

        # Load weights shard by shard
        all_weights = {}
        for shard_file in sorted(shard_weights.keys()):
            shard_path = self.model_dir / shard_file
            print(f"Loading {shard_file}...")
            shard_data = mx.load(str(shard_path))
            all_weights.update(shard_data)

        # Map weights to model parameters
        self._load_weights_from_dict(all_weights)

    def _load_weights_from_dict(self, weights_dict: Dict[str, mx.array]):
        """Map safetensors weights to model parameters."""
        # Load embedding
        if self.embedding is not None and 'backbone.embeddings.weight' in weights_dict:
            self.embedding.weight = weights_dict['backbone.embeddings.weight']

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
                        setattr(obj, parts[-1], weights_dict[safetensors_key])

        # Load output norm
        if self.out_norm is not None and 'backbone.out_norm.weight' in weights_dict:
            self.out_norm.weight = weights_dict['backbone.out_norm.weight']

        # Load lm_head
        if self.lm_head is not None and 'lm_head.weight' in weights_dict:
            self.lm_head.weight = weights_dict['lm_head.weight']

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
        }


__all__ = ['WiredxLSTM']
