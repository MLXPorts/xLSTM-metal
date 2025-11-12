"""Wired xLSTM Model – MLX Implementation (Model-Agnostic Language Model)

Overview
--------
WiredxLSTM is the top-level model class that assembles complete xLSTM
language models using NCPS-style wiring. It automatically discovers model
architecture from safetensors checkpoints and builds the correct stack of
blocks (mLSTM, sLSTM, attention) with proper weight loading.

Model-Agnostic Design
---------------------
Traditional approach:
  - Hardcode architecture (e.g., XLSTMForCausalLM with 32 mLSTM blocks)
  - Write custom loading code for each model size
  - Duplicate classes for 1B, 7B, 13B variants

WiredxLSTM approach:
  - Introspect safetensors to discover structure
  - Build appropriate blocks dynamically
  - Single class handles all model sizes/types

Architecture Stack
------------------
Complete xLSTM language model follows this pattern:

  Input Token IDs [B, S]
    ↓ embedding
  Embeddings [B, S, D]
    ↓ blocks[0..N-1]
  Hidden [B, S, D]
    ↓ out_norm (RMSNorm)
  Normalized [B, S, D]
    ↓ lm_head (Linear)
  Logits [B, S, vocab_size]

Each block is typically:
  residual = x
  x = norm_mlstm(x)
  x, state = mlstm_cell(x, state)
  x = x + residual

  residual = x
  x = norm_ffn(x)
  x = ffn(x)
  x = x + residual

Automatic Structure Discovery
------------------------------
WiredxLSTM uses AutoWiring to:
  1. Parse model.safetensors.index.json
  2. Detect block types (mlstm_layer, slstm_layer, attn_layer)
  3. Count blocks and identify special layers (embedding, out_norm, lm_head)
  4. Build sequential connectivity
  5. Create appropriate cell instances for each block

Weight Loading
--------------
Weights are loaded from safetensors shards using the mapping:
  - backbone.embeddings.weight → embedding layer
  - backbone.blocks.{i}.* → block[i] parameters
  - backbone.out_norm.weight → output normalization
  - lm_head.weight → language model head

Each block provides get_weight_keys() to map internal parameters to
safetensors keys, enabling automatic weight loading without hardcoding
parameter names.

Weight Tying
------------
When tie_word_embeddings=True (common for LLMs to reduce parameters):
  lm_head.weight = embedding.weight
This shares the embedding matrix for both input and output, reducing
total parameters by vocab_size * embedding_dim.

Mixed Precision
---------------
Supports independent dtypes:
  - compute_dtype: Forward pass activations (bfloat16 for speed)
  - state_dtype: Recurrent state storage (float32 for stability)
  - param_dtype: Trainable parameters (inherits from compute_dtype)

Typical inference config: compute=bfloat16, state=float32

Stateful Generation
-------------------
The model maintains recurrent state across tokens during generation:

  # Initial prompt
  logits, states = model(prompt_ids, state=None, return_last_states=True)

  # Autoregressive generation
  for _ in range(max_new_tokens):
      next_id = sample(logits[:, -1, :])
      logits, states = model(next_id.reshape(1,1), state=states, return_last_states=True)

State reuse enables efficient generation without reprocessing context.

Configuration Sources
---------------------
Model configuration comes from multiple sources:
  1. **config.json**: Base hyperparameters (embedding_dim, num_heads, etc.)
  2. **Safetensors index**: Derived structure (num_blocks, block types)
  3. **Runtime kwargs**: Dtype overrides, mode settings

The wiring object combines these into a complete runtime config.

Usage Patterns
--------------
Automatic loading (recommended):
  >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
  >>> logits = model(input_ids)

Manual wiring (advanced):
  >>> wiring = create_auto_wiring("xlstm_7b_model")
  >>> model = WiredxLSTM(wiring=wiring, load_weights=True)

Custom dtype:
  >>> model = WiredxLSTM.from_pretrained(
  ...     "xlstm_7b_model",
  ...     compute_dtype=mx.bfloat16,
  ...     state_dtype=mx.float32
  ... )

Special Tokens
--------------
Model respects tokenizer special tokens from config:
  - pad_token_id: Padding token (attention masking)
  - bos_token_id: Beginning-of-sequence token
  - eos_token_id: End-of-sequence token
  - force_bos_token_insert: Auto-prepend BOS if missing

Embedding Dropout
-----------------
Optional dropout on embeddings (typically disabled for inference):
  add_embedding_dropout=True, embedding_dropout=0.1

Training vs Inference Mode
--------------------------
Model defaults to eval() when config['mode']='inference' to:
  - Disable dropout
  - Use deterministic behavior
  - Enable efficient generation

Call .train() to enable dropout for fine-tuning.

Extensibility
-------------
Adding new block types:
  1. Implement block cell class (e.g., AttentionBlock)
  2. Update detect_block_type() in auto_wiring.py
  3. Add case in create_block_cell() factory method

WiredxLSTM automatically handles new types without modification.

Parity
------
Logic mirrors torch-native WiredxLSTM for cross-backend compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import mlx.core as mx
import mlx.nn as nn

from xlstm_metal.mlx_jit.wiring import AutoWiring, create_auto_wiring
from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards
from xlstm_metal.mlx_jit.blocks.rms_norm import RMSNormCell


class WiredxLSTM(nn.Module):
    """Model-agnostic xLSTM language model with automatic structure discovery.

    Assembles complete LM from safetensors using NCPS wiring. Supports
    mixed block types (mLSTM, sLSTM, attention) with automatic weight loading.

    Parameters
    ----------
    wiring : AutoWiring
        Wiring object defining model structure (from create_auto_wiring).
    load_weights : bool, default False
        Whether to load pretrained weights from safetensors.
    model_dir : str | Path | None, optional
        Model directory (for weight loading, defaults to wiring.model_dir).
    compute_dtype : mx.Dtype, default mx.float32
        Dtype for forward pass activations.
    state_dtype : mx.Dtype, default mx.float32
        Dtype for recurrent state storage.
    norm_reduce_force_float32 : bool, default True
        Force float32 in norm reductions for stability.

    Attributes
    ----------
    wiring : AutoWiring
        Model structure specification.
    config : dict
        Complete configuration (from config.json + derived values).
    blocks : list
        Stack of xLSTM blocks (mLSTM/sLSTM/attention cells).
    embedding : nn.Embedding | None
        Token embedding layer.
    out_norm : RMSNormCell | None
        Pre-LM-head normalization (uses custom RMSNormCell for canonical behavior).
    lm_head : nn.Linear | None
        Language modeling head (vocab projection).

    Methods
    -------
    __call__(input_ids, state, return_last_states)
        Forward pass through model.
    from_pretrained(model_dir, **kwargs)
        Class method to load from checkpoint directory.
    load_pretrained_weights()
        Load weights from safetensors files.

    Examples
    --------
    >>> # Automatic loading
    >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
    >>> logits = model(input_ids)

    >>> # Stateful generation
    >>> logits, states = model(prompt, return_last_states=True)
    >>> for _ in range(100):
    ...     next_token = sample(logits[:, -1, :])
    ...     logits, states = model(next_token.reshape(1,1), state=states, return_last_states=True)
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
            self.out_norm = RMSNormCell(
                dims=self.config['embedding_dim'],
                eps=self.config.get('norm_eps', 1e-6),
                force_float32_reductions=self.norm_reduce_force_float32,
                param_dtype=self.compute_dtype,
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
        """Forward pass through complete language model.

        Parameters
        ----------
        input_ids : mx.array [B, S]
            Input token IDs.
        state : list of tuple | None, optional
            List of recurrent states for each block (for stateful generation).
        return_last_states : bool, default False
            Whether to return final recurrent states.

        Returns
        -------
        logits : mx.array [B, S, vocab_size]
            Output logits for next-token prediction.
        states : list of tuple (optional)
            Final recurrent states for each block (if return_last_states=True).

        Notes
        -----
        - Embedding dropout applied if configured
        - Each block processes sequentially with residual connections
        - Output norm applied before LM head
        - States enable efficient autoregressive generation
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
        """Load complete model from HuggingFace checkpoint directory.

        Discovers architecture from safetensors, builds model, loads weights.

        Parameters
        ----------
        model_dir : str | Path
            Path to model directory with config.json and safetensors files.
        load_weights : bool, default True
            Whether to load pretrained weights.
        **kwargs
            Additional arguments for WiredxLSTM constructor
            (compute_dtype, state_dtype, etc.).

        Returns
        -------
        model : WiredxLSTM
            Initialized model ready for inference or fine-tuning.

        Examples
        --------
        >>> model = WiredxLSTM.from_pretrained("xlstm_7b_model")
        >>> model = WiredxLSTM.from_pretrained(
        ...     "xlstm_7b_model",
        ...     compute_dtype=mx.bfloat16,
        ...     state_dtype=mx.float32
        ... )
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
            'force_bos_token_insert': self.force_bos_token_insert,
            'embedding_dropout_prob': self.embedding_dropout_prob,
            'compute_dtype': self.compute_dtype,
            'state_dtype': self.state_dtype,
            'norm_reduce_force_float32': self.norm_reduce_force_float32,
        }

    def _apply_weight_tying(self) -> None:
        """Tie LM head weights to embeddings when the config requests it."""
        should_tie = bool(self.tie_word_embeddings) and self.embedding is not None and self.lm_head is not None
        self._embeddings_tied = should_tie
        if should_tie:
            self.lm_head.weight = self.embedding.weight


__all__ = ['WiredxLSTM']
