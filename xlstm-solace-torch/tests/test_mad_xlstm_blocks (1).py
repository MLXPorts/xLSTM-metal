#!/usr/bin/env python
"""Test xLSTM blocks as MAD-style layers."""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from xlstm_solace_torch.mad.registry import layer_registry
from xlstm_solace_torch.mad.model.layers import MLSTMBlock, GatedFFN


def test_mlstm_block_in_registry():
    """Test that MLSTMBlock is registered in layer_registry."""
    assert 'mlstm' in layer_registry
    assert layer_registry['mlstm']['module'] == MLSTMBlock
    assert layer_registry['mlstm']['shorthand'] == 'mL'


def test_gated_ffn_in_registry():
    """Test that GatedFFN is registered in layer_registry."""
    assert 'xlstm-gated-ffn' in layer_registry
    assert layer_registry['xlstm-gated-ffn']['module'] == GatedFFN
    assert layer_registry['xlstm-gated-ffn']['shorthand'] == 'xF'


def test_mlstm_block_forward():
    """Test MLSTMBlock forward pass."""
    dim = 64
    batch_size = 2
    seq_len = 10

    block = MLSTMBlock(
        dim=dim,
        num_heads=4,
        conv_kernel_size=4,
        proj_factor=2.0,
        use_causal_conv=True,
        use_learnable_skip=True,
        backend='pytorch',
        bias=False,
        dropout=0.0
    )

    x = torch.randn(batch_size, seq_len, dim)
    y = block(x)

    assert y.shape == (batch_size, seq_len, dim)


def test_gated_ffn_forward():
    """Test GatedFFN forward pass."""
    dim = 64
    batch_size = 2
    seq_len = 10

    ffn = GatedFFN(
        dim=dim,
        proj_factor=2.667,
        round_up_to=64,
        bias=False,
        dropout=0.0
    )

    x = torch.randn(batch_size, seq_len, dim)
    y = ffn(x)

    assert y.shape == (batch_size, seq_len, dim)


def test_mlstm_block_without_conv():
    """Test MLSTMBlock without causal convolution."""
    dim = 64
    batch_size = 2
    seq_len = 10

    block = MLSTMBlock(
        dim=dim,
        num_heads=4,
        use_causal_conv=False,
        backend='pytorch'
    )

    x = torch.randn(batch_size, seq_len, dim)
    y = block(x)

    assert y.shape == (batch_size, seq_len, dim)


def test_mlstm_block_without_skip():
    """Test MLSTMBlock without learnable skip."""
    dim = 64
    batch_size = 2
    seq_len = 10

    block = MLSTMBlock(
        dim=dim,
        num_heads=4,
        use_learnable_skip=False,
        backend='pytorch'
    )

    x = torch.randn(batch_size, seq_len, dim)
    y = block(x)

    assert y.shape == (batch_size, seq_len, dim)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
