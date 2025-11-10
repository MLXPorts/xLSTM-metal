#!/usr/bin/env python
"""
Weight loader for xLSTM-7B NPZ weights → MAD blocks

Maps the flattened NPZ weight structure to MAD block hierarchy.
Supports both standalone xLSTMBlock lists and WiredMADModel instances.
"""

from typing import Dict, TYPE_CHECKING

import mlx.core as mx

from ..models.xlstm_7b_model import xLSTM7BCell


if TYPE_CHECKING:
    from xlstm_metal.mlx_jit.wiring import WiredMADModel


def load_npz_weights_to_block(npz_weights: Dict[str, mx.array], block_idx: int, block: xLSTM7BCell):
    """
    Load weights from NPZ into an xLSTM7BCell.

    NPZ structure (from convert_hf_to_mlx.py):
        blocks.{i}.W_q.weight -> xlstm.q.weight
        blocks.{i}.W_k.weight -> xlstm.k.weight
        blocks.{i}.W_v.weight -> xlstm.v.weight
        blocks.{i}.W_i.weight/bias -> xlstm.igate_preact.weight/bias
        blocks.{i}.W_f.weight/bias -> xlstm.fgate_preact.weight/bias
        blocks.{i}.W_o.weight -> xlstm.out_proj.weight
        blocks.{i}.mhln.weight -> xlstm.multihead_norm.weight
        blocks.{i}.norm.weight -> xlstm_norm.weight
        blocks.{i}.norm2.weight -> ffn_norm.weight
        blocks.{i}.up_l_proj.weight -> ffn.proj_up (gate part)
        blocks.{i}.up_r_proj.weight -> ffn.proj_up (up part)
        blocks.{i}.down_proj.weight -> ffn.proj_down.weight

    Args:
        npz_weights: Loaded NPZ weights dict
        block_idx: Block index (0-31 for xLSTM-7B)
        block: xLSTMBlock to load weights into

    Note: The NPZ uses `up_l_proj` and `up_r_proj` for gate+up projections,
          but our GatedFFN uses a single `proj_up` that outputs 2x dims.
          We need to concatenate them.
    """
    prefix = f"blocks.{block_idx}"

    # mLSTM layer weights
    mlstm_mappings = {
        f"{prefix}.W_q.weight": ("xlstm", "q", "weight"),
        f"{prefix}.W_k.weight": ("xlstm", "k", "weight"),
        f"{prefix}.W_v.weight": ("xlstm", "v", "weight"),
        f"{prefix}.W_i.weight": ("xlstm", "igate_preact", "weight"),
        f"{prefix}.W_i.bias": ("xlstm", "igate_preact", "bias"),
        f"{prefix}.W_f.weight": ("xlstm", "fgate_preact", "weight"),
        f"{prefix}.W_f.bias": ("xlstm", "fgate_preact", "bias"),
        f"{prefix}.W_o.weight": ("xlstm", "out_proj", "weight"),
    }

    for npz_key, (module, submodule, param) in mlstm_mappings.items():
        if npz_key in npz_weights:
            target = getattr(getattr(block, module), submodule)
            setattr(target, param, npz_weights[npz_key])
        else:
            print(f"Warning: Missing weight {npz_key}")

    # Multi-head layer norm
    # NPZ has shape [4096] (flattened), we need [8, 512] (num_heads, head_dim)
    mhln_key = f"{prefix}.mhln.weight"
    if mhln_key in npz_weights:
        mhln_weight = npz_weights[mhln_key]
        # Reshape from [v_dim] to [num_heads, head_dim]
        num_heads = block.xlstm.config.num_heads
        head_dim = block.xlstm.config.head_dim
        block.xlstm.multihead_norm.weight = mhln_weight.reshape(num_heads, head_dim)
    else:
        print(f"Warning: Missing weight {mhln_key}")

    # Pre-normalization weights
    norm_key = f"{prefix}.norm.weight"
    if norm_key in npz_weights:
        block.xlstm_norm.weight = npz_weights[norm_key]
    else:
        print(f"Warning: Missing weight {norm_key}")

    norm2_key = f"{prefix}.norm2.weight"
    if norm2_key in npz_weights:
        block.ffn_norm.weight = npz_weights[norm2_key]
    else:
        print(f"Warning: Missing weight {norm2_key}")

    # FFN weights - concatenate up_l_proj and up_r_proj
    up_l_key = f"{prefix}.up_l_proj.weight"
    up_r_key = f"{prefix}.up_r_proj.weight"
    if up_l_key in npz_weights and up_r_key in npz_weights:
        # NPZ has separate gate and up projections
        # Our GatedFFN.proj_up outputs 2x dims: [2*proj_up_dim, embedding_dim]
        # Concatenate along output dimension (dim 0)
        up_l = npz_weights[up_l_key]  # [proj_up_dim, embedding_dim]
        up_r = npz_weights[up_r_key]  # [proj_up_dim, embedding_dim]
        block.ffn.proj_up.weight = mx.concatenate([up_l, up_r])
    else:
        print(f"Warning: Missing FFN up projections")

    down_key = f"{prefix}.down_proj.weight"
    if down_key in npz_weights:
        block.ffn.proj_down.weight = npz_weights[down_key]
    else:
        print(f"Warning: Missing weight {down_key}")


def load_xLSTM_7b_weights(npz_path: str, blocks: list[xLSTMBlock]):
    """
    Load xLSTM-7B weights from NPZ file into list of xLSTMBlocks.

    Args:
        npz_path: Path to xlstm_7b_mlx_converted.npz
        blocks: List of 32 xLSTMBlock instances

    Returns:
        embedding_weight: [vocab_size, embedding_dim]
        head_weight: [vocab_size, embedding_dim]
    """
    print(f"Loading weights from {npz_path}...")
    weights = mx.load(npz_path)

    print(f"Loaded {len(weights)} weight tensors")

    # Load block weights
    assert len(blocks) == 32, f"Expected 32 blocks, got {len(blocks)}"

    for i, block in enumerate(blocks):
        print(f"Loading block {i}...")
        load_npz_weights_to_block(weights, i, block)

    # Extract embedding and head weights
    embedding_weight = weights.get("embedding.weight")
    head_weight = weights.get("head.W")

    if embedding_weight is None:
        raise ValueError("Missing embedding.weight in NPZ")
    if head_weight is None:
        raise ValueError("Missing head.W in NPZ")

    print(f"✓ Loaded all weights")
    print(f"  Embedding: {embedding_weight.shape}")
    print(f"  Head: {head_weight.shape}")

    return embedding_weight, head_weight


def load_weights_into_wired_model(npz_path: str, model: "WiredMADModel"):
    """
    Load xLSTM-7B weights from NPZ file into a WiredMADModel.

    This function handles the MAD wiring structure where blocks are named
    'xlstm_0', 'xlstm_1', etc., and also loads embedding and lm_head weights.

    Args:
        npz_path: Path to xlstm_7b_mlx_converted.npz
        model: WiredMADModel instance with xLSTM-7B wiring

    The model is expected to have blocks named:
        - 'embedding': Token embedding
        - 'xlstm_0' through 'xlstm_31': xLSTM blocks
        - 'final_norm': Final RMSNorm
        - 'lm_head': Language model head
    """
    print(f"Loading weights into WiredMADModel from {npz_path}...")
    weights = mx.load(npz_path)

    print(f"Loaded {len(weights)} weight tensors")

    # Load xLSTM block weights
    num_blocks = sum(1 for name in model.blocks.keys() if name.startswith('xlstm_'))
    print(f"Found {num_blocks} xLSTM blocks in model")

    for i in range(num_blocks):
        block_name = f'xlstm_{i}'
        if block_name in model.blocks:
            print(f"Loading weights into {block_name}...")
            block = model.blocks[block_name]
            load_npz_weights_to_block(weights, i, block)
        else:
            print(f"Warning: Block {block_name} not found in model")

    # Load embedding weights
    if 'embedding' in model.blocks:
        embedding_weight = weights.get("embedding.weight")
        if embedding_weight is not None:
            model.blocks['embedding'].weight = embedding_weight
            print(f"✓ Loaded embedding weights: {embedding_weight.shape}")
        else:
            print("Warning: embedding.weight not found in NPZ")
    else:
        print("Warning: 'embedding' block not found in model")

    # Load out_norm weights (CRITICAL - backbone.out_norm)
    if 'out_norm' in model.blocks:
        out_norm_weight = weights.get("out_norm.weight")
        if out_norm_weight is not None:
            model.blocks['out_norm'].weight = out_norm_weight
            print(f"✓ Loaded out_norm weights: {out_norm_weight.shape}")
        else:
            print("Warning: out_norm.weight not found in NPZ")
    else:
        print("Warning: 'out_norm' block not found in model")

    # Load lm_head weights
    if 'lm_head' in model.blocks:
        head_weight = weights.get("head.W")
        if head_weight is not None:
            model.blocks['lm_head'].weight = head_weight
            print(f"✓ Loaded lm_head weights: {head_weight.shape}")
        else:
            print("Warning: head.W not found in NPZ")
    else:
        print("Warning: 'lm_head' block not found in model")

    print(f"\n✅ Successfully loaded all pretrained weights into WiredMADModel")
