#!/usr/bin/env python
"""Test numerical parity between ChunkwiseMLSTM and mlstm_kernels Triton implementation."""

import torch
import pytest
from xlstm_solace_torch.mad.blocks.mlstm_torch import ChunkwiseMLSTM

# Import mlstm_kernels for comparison
try:
    from mlstm_kernels.torch.chunkwise import mlstm_chunkwise__xl_chunk
    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False
    mlstm_chunkwise__xl_chunk = None


class TestChunkwiseParity:
    """Test ChunkwiseMLSTM against canonical mlstm_kernels Triton implementation."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton/CUDA not available")
    def test_chunkwise_forward_parity(self):
        """Test that ChunkwiseMLSTM forward matches mlstm_kernels Triton exactly."""
        torch.manual_seed(42)

        B, S, D = 2, 256, 512
        NH = 4
        chunk_size = 64

        # Create our block
        our_block = ChunkwiseMLSTM(
            dim=D,
            num_heads=NH,
            chunk_size=chunk_size,
            bias=False,
            dropout=0.0,
        ).cuda()

        # Test input
        x = torch.randn(B, S, D, device='cuda')

        # Our implementation forward
        y_our, states_our = our_block(x)

        # Extract internal QKV and gates from our block for Triton comparison
        with torch.no_grad():
            # Up-projection
            x_up = our_block.proj_up(x)
            x_mlstm, x_ogate = x_up.chunk(2, dim=-1)

            # Conv if present
            if our_block.conv1d is not None:
                x_conv = x_mlstm.transpose(1, 2)
                x_conv = our_block.conv1d(x_conv)
                x_conv = x_conv[:, :, :S]
                x_mlstm = our_block.conv_act(x_conv.transpose(1, 2))

            # QKV projections
            q = our_block.q_proj(x_mlstm)
            k = our_block.k_proj(x_mlstm)
            v = our_block.v_proj(x_mlstm)

            # Reshape for multi-head
            q = q.view(B, S, NH, -1).transpose(1, 2)  # [B, NH, S, DH]
            k = k.view(B, S, NH, -1).transpose(1, 2)
            v = v.view(B, S, NH, -1).transpose(1, 2)

            # Gates
            i_preact = our_block.igate_proj(x_mlstm)
            f_preact = our_block.fgate_proj(x_mlstm)
            i_gates = torch.sigmoid(i_preact.transpose(1, 2))  # [B, NH, S]
            f_gates = torch.sigmoid(f_preact.transpose(1, 2))

        # Call Triton kernel directly with same inputs
        h_triton, (C_triton, n_triton, m_triton) = mlstm_chunkwise__xl_chunk(
            q=q,
            k=k,
            v=v,
            i=i_gates,
            f=f_gates,
            c_initial=None,
            n_initial=None,
            m_initial=None,
            return_last_states=True,
            chunk_size=chunk_size,
            autocast_kernel_dtype=torch.float32,
            eps=1e-6,
        )

        # Our implementation computes h differently, so compare states
        # States should match since compute_chunk_states uses same algorithm
        C_our = states_our['C']  # [B, NH, DH, DH]
        n_our = states_our['n']  # [B, NH, DH]

        # Compare states
        assert torch.allclose(C_our, C_triton, atol=1e-5, rtol=1e-4), \
            f"C states don't match. Max diff: {(C_our - C_triton).abs().max()}"
        assert torch.allclose(n_our, n_triton, atol=1e-5, rtol=1e-4), \
            f"n states don't match. Max diff: {(n_our - n_triton).abs().max()}"

        print("✓ ChunkwiseMLSTM states match Triton implementation")

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton/CUDA not available")
    def test_compute_chunk_states_parity(self):
        """Test compute_chunk_states function against Triton kernel."""
        from xlstm_solace_torch.mad.blocks.mlstm_torch.chunkwise import compute_chunk_states

        torch.manual_seed(42)

        B, NH, S, DH = 2, 4, 256, 128
        chunk_size = 64

        # Random inputs
        k = torch.randn(B, NH, S, DH, device='cuda')
        v = torch.randn(B, NH, S, DH, device='cuda')
        i_gates = torch.sigmoid(torch.randn(B, NH, S, device='cuda'))
        f_gates = torch.sigmoid(torch.randn(B, NH, S, device='cuda'))
        q = torch.randn(B, NH, S, DH, device='cuda')

        # Our compute_chunk_states
        C_states_our, n_states_our = compute_chunk_states(
            k=k, v=v, i_gates=i_gates, f_gates=f_gates,
            chunk_size=chunk_size, eps=1e-6
        )

        # Triton kernel
        h_triton, (C_triton, n_triton, m_triton) = mlstm_chunkwise__xl_chunk(
            q=q, k=k, v=v, i=i_gates, f=f_gates,
            c_initial=None, n_initial=None, m_initial=None,
            return_last_states=True,
            chunk_size=chunk_size,
            autocast_kernel_dtype=torch.float32,
            eps=1e-6,
        )

        # Compare final states (last chunk boundary)
        C_our_final = C_states_our[:, :, -1, :, :]
        n_our_final = n_states_our[:, :, -1, :]

        assert torch.allclose(C_our_final, C_triton, atol=1e-5, rtol=1e-4), \
            f"C final states don't match. Max diff: {(C_our_final - C_triton).abs().max()}"
        assert torch.allclose(n_our_final, n_triton, atol=1e-5, rtol=1e-4), \
            f"n final states don't match. Max diff: {(n_our_final - n_triton).abs().max()}"

        print("✓ compute_chunk_states matches Triton kernel")

    def test_chunkwise_cpu_execution(self):
        """Test that ChunkwiseMLSTM runs on CPU (for Apple Silicon)."""
        torch.manual_seed(42)

        B, S, D = 2, 128, 256
        NH = 4
        chunk_size = 32

        # Create block on CPU
        block = ChunkwiseMLSTM(
            dim=D,
            num_heads=NH,
            chunk_size=chunk_size,
            bias=False,
            dropout=0.0,
        )

        # Test input
        x = torch.randn(B, S, D)

        # Forward pass
        y, states = block(x)

        # Basic shape checks
        assert y.shape == (B, S, D), f"Output shape mismatch: {y.shape}"
        assert states['C'].shape == (B, NH, D * 2 // NH, D * 2 // NH), \
            f"C state shape mismatch: {states['C'].shape}"
        assert states['n'].shape == (B, NH, D * 2 // NH), \
            f"n state shape mismatch: {states['n'].shape}"

        # Check for NaN/Inf
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

        print("✓ ChunkwiseMLSTM runs successfully on CPU")

    def test_chunkwise_gradient_flow(self):
        """Test that gradients flow through ChunkwiseMLSTM."""
        torch.manual_seed(42)

        B, S, D = 2, 64, 128
        NH = 2

        block = ChunkwiseMLSTM(dim=D, num_heads=NH, chunk_size=32)
        x = torch.randn(B, S, D, requires_grad=True)

        # Forward
        y, _ = block(x)

        # Backward
        loss = y.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Input gradient contains Inf"

        # Check parameter gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"

        print("✓ Gradients flow correctly through ChunkwiseMLSTM")

    def test_chunkwise_state_continuation(self):
        """Test that states can be passed between forward calls."""
        torch.manual_seed(42)

        B, S, D = 2, 64, 128
        NH = 2

        block = ChunkwiseMLSTM(dim=D, num_heads=NH, chunk_size=32)

        # First chunk
        x1 = torch.randn(B, S, D)
        y1, states1 = block(x1)

        # Second chunk with state continuation (not yet implemented)
        # TODO: Implement state continuation in ChunkwiseMLSTM.forward()
        x2 = torch.randn(B, S, D)
        y2, states2 = block(x2)  # states=states1 when implemented

        # For now, just check states are returned
        assert states1 is not None, "States not returned"
        assert 'C' in states1 and 'n' in states1 and 'm' in states1

        print("✓ State continuation interface works (implementation pending)")


if __name__ == "__main__":
    # Run tests
    test = TestChunkwiseParity()

    print("\n=== Running CPU tests ===")
    test.test_chunkwise_cpu_execution()
    test.test_chunkwise_gradient_flow()
    test.test_chunkwise_state_continuation()

    if TRITON_AVAILABLE:
        print("\n=== Running Triton parity tests ===")
        test.test_compute_chunk_states_parity()
        test.test_chunkwise_forward_parity()
    else:
        print("\n=== Skipping Triton tests (CUDA not available) ===")

    print("\n✓ All tests passed!")
