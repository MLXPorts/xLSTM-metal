#!/usr/bin/env python
"""Test numerical parity between our MAD blocks and canonical xLSTM."""

import torch
import pytest
import sys
from pathlib import Path

# Add both repos to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, "/Volumes/stuff/Projects/xlstm")

from xlstm_solace_torch.mad.blocks.mlstm_torch.cell import mLSTMCell, mLSTMCellConfig
from xlstm_solace_torch.mad.blocks.mlstm_torch.layer import mLSTMLayer, mLSTMLayerConfig

# Canonical imports
from xlstm.blocks.mlstm.cell import mLSTMCell as CanonicalCell, mLSTMCellConfig as CanonicalCellConfig
from xlstm.blocks.mlstm.layer import mLSTMLayer as CanonicalLayer, mLSTMLayerConfig as CanonicalLayerConfig


class TestmLSTMCellParity:
    """Compare our mLSTM cell against canonical."""

    def test_cell_forward_parity(self):
        """Test that cell forward matches canonical exactly."""
        torch.manual_seed(42)
        B, S, D, NH = 2, 16, 128, 4
        
        config = mLSTMCellConfig(
            context_length=S,
            embedding_dim=D,
            num_heads=NH
        )
        
        canonical_config = CanonicalCellConfig(
            context_length=S,
            embedding_dim=D,
            num_heads=NH
        )
        
        # Create cells
        our_cell = mLSTMCell(canonical_config)
        canonical_cell = CanonicalCell(canonical_config)
        
        # Copy weights to ensure same initialization
        canonical_cell.load_state_dict(our_cell.state_dict())
        
        # Same inputs
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)
        
        # Forward
        our_output = our_cell(q, k, v)
        canonical_output = canonical_cell(q, k, v)
        
        # Check numerical parity
        assert torch.allclose(our_output, canonical_output, atol=1e-6, rtol=1e-5), \
            f"Max diff: {(our_output - canonical_output).abs().max()}"

    def test_cell_backend_parity(self):
        """Test parallel_stabilized_simple matches canonical."""
        from xlstm_solace_torch.mad.blocks.mlstm_torch.backends import parallel_stabilized_simple
        from xlstm.blocks.mlstm.backends import parallel_stabilized_simple as canonical_backend
        
        torch.manual_seed(42)
        B, NH, S, DH = 2, 4, 16, 32
        
        q = torch.randn(B, NH, S, DH)
        k = torch.randn(B, NH, S, DH)
        v = torch.randn(B, NH, S, DH)
        igate = torch.randn(B, NH, S, 1)
        fgate = torch.randn(B, NH, S, 1)
        
        our_output = parallel_stabilized_simple(q, k, v, igate, fgate)
        canonical_output = canonical_backend(q, k, v, igate, fgate)
        
        assert torch.allclose(our_output, canonical_output, atol=1e-6, rtol=1e-5), \
            f"Max diff: {(our_output - canonical_output).abs().max()}"

    def test_cell_step_parity(self):
        """Test single-step recurrent forward matches canonical."""
        from xlstm_solace_torch.mad.blocks.mlstm_torch.backends import recurrent_step_stabilized_simple
        from xlstm.blocks.mlstm.backends import recurrent_step_stabilized_simple as canonical_step
        
        torch.manual_seed(42)
        B, NH, DH = 2, 4, 32
        
        # Initial states
        c_state = torch.randn(B, NH, DH, DH)
        n_state = torch.randn(B, NH, DH, 1)
        m_state = torch.randn(B, NH, 1, 1)
        
        # Single timestep
        q = torch.randn(B, NH, DH).unsqueeze(2)
        k = torch.randn(B, NH, DH).unsqueeze(2)
        v = torch.randn(B, NH, DH).unsqueeze(2)
        igate = torch.randn(B, NH, 1, 1)
        fgate = torch.randn(B, NH, 1, 1)

        # Clone tensors since backends use in-place squeeze_()
        our_h, our_states = recurrent_step_stabilized_simple(
            c_state.clone(), n_state.clone(), m_state.clone(),
            q.clone(), k.clone(), v.clone(), igate.clone(), fgate.clone()
        )
        canonical_h, canonical_states = canonical_step(
            c_state.clone(), n_state.clone(), m_state.clone(),
            q.clone(), k.clone(), v.clone(), igate.clone(), fgate.clone()
        )
        
        assert torch.allclose(our_h, canonical_h, atol=1e-6, rtol=1e-5)
        assert torch.allclose(our_states[0], canonical_states[0], atol=1e-6, rtol=1e-5)
        assert torch.allclose(our_states[1], canonical_states[1], atol=1e-6, rtol=1e-5)
        assert torch.allclose(our_states[2], canonical_states[2], atol=1e-6, rtol=1e-5)


class TestmLSTMLayerParity:
    """Compare our mLSTM layer against canonical."""

    def test_layer_forward_parity(self):
        """Test that layer forward matches canonical exactly."""
        torch.manual_seed(42)
        B, S, D = 2, 32, 128
        
        config = mLSTMLayerConfig(
            embedding_dim=D,
            context_length=S,
            num_heads=4,
            conv1d_kernel_size=4,
        )
        
        canonical_config = CanonicalLayerConfig(
            embedding_dim=D,
            context_length=S,
            num_heads=4,
            conv1d_kernel_size=4,
        )
        
        # Create layers
        our_layer = mLSTMLayer(canonical_config)
        canonical_layer = CanonicalLayer(canonical_config)
        
        # Copy weights
        canonical_layer.load_state_dict(our_layer.state_dict())
        
        # Same input
        x = torch.randn(B, S, D)
        
        # Forward
        our_output = our_layer(x)
        canonical_output = canonical_layer(x)
        
        # Check numerical parity
        max_diff = (our_output - canonical_output).abs().max().item()
        assert torch.allclose(our_output, canonical_output, atol=1e-5, rtol=1e-4), \
            f"Max diff: {max_diff}"

    def test_layer_backward_parity(self):
        """Test that gradients match canonical."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64
        
        config = CanonicalLayerConfig(
            embedding_dim=D,
            context_length=S,
            num_heads=4,
        )
        
        our_layer = mLSTMLayer(config)
        canonical_layer = CanonicalLayer(config)
        canonical_layer.load_state_dict(our_layer.state_dict())
        
        x = torch.randn(B, S, D, requires_grad=True)
        x_canonical = x.clone().detach().requires_grad_(True)
        
        # Forward + backward
        our_output = our_layer(x)
        canonical_output = canonical_layer(x_canonical)
        
        loss_ours = our_output.sum()
        loss_canonical = canonical_output.sum()
        
        loss_ours.backward()
        loss_canonical.backward()
        
        # Check gradient parity
        assert torch.allclose(x.grad, x_canonical.grad, atol=1e-5, rtol=1e-4), \
            f"Max grad diff: {(x.grad - x_canonical.grad).abs().max()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
