"""
Pytest tests for xLSTM NCPS Wiring

Tests the Neural Circuit Policies (NCPS) wiring implementation for xLSTM.
"""

import pytest
import mlx.core as mx
from xlstm_metal.mlx_blocks.wiring import create_xlstm_wiring, xLSTMWiring, AutoNCPxLSTMWiring
from xlstm_metal.mlx_blocks.wiring import Wiring


@pytest.fixture
def xlstm_config():
    """Standard xLSTM-7B configuration."""
    return {
        'embedding_dim': 4096,
        'num_heads': 8,
        'num_blocks': 32,
        'vocab_size': 50304,
        'qk_dim_factor': 0.5,
        'v_dim_factor': 1.0,
        'ffn_proj_factor': 2.671875,
        'gate_soft_cap': 15.0,
        'norm_eps': 1e-6,
        'output_logit_soft_cap': 30.0
    }


@pytest.fixture
def xlstm_wiring(xlstm_config):
    """Create xLSTM wiring from config."""
    return create_xlstm_wiring(xlstm_config)


class TestWiringCreation:
    """Test wiring creation and initialization."""

    def test_create_xlstm_wiring(self, xlstm_config):
        """Test creating xLSTM wiring from config."""
        wiring = create_xlstm_wiring(xlstm_config)

        assert isinstance(wiring, Wiring)
        assert isinstance(wiring, xLSTMWiring)
        assert wiring.units == xlstm_config['num_blocks']
        assert wiring.num_layers == 1

    def test_wiring_neurons(self, xlstm_wiring, xlstm_config):
        """Test wiring neuron structure."""
        layer_0_neurons = xlstm_wiring.get_neurons_of_layer(0)

        assert len(layer_0_neurons) == xlstm_config['num_blocks']
        assert layer_0_neurons == list(range(xlstm_config['num_blocks']))

    def test_neuron_types(self, xlstm_wiring):
        """Test neuron type queries."""
        for i in range(xlstm_wiring.units):
            neuron_type = xlstm_wiring.get_type_of_neuron(i)
            assert neuron_type == "xlstm"


class TestWiringFunctionality:
    """Test wiring functionality."""

    def test_build_wiring(self, xlstm_wiring):
        """Test building wiring with input dimensions."""
        input_dim = 4096

        assert not xlstm_wiring.is_built()

        xlstm_wiring.build(input_dim)

        assert xlstm_wiring.is_built()
        assert xlstm_wiring.input_dim == input_dim

    def test_visualize(self, xlstm_wiring):
        """Test wiring visualization."""
        viz = xlstm_wiring.visualize()

        assert isinstance(viz, str)
        assert "NCPS Neural Circuit" in viz
        assert "xlstm" in viz

    def test_get_config(self, xlstm_wiring):
        """Test getting wiring configuration."""
        config = xlstm_wiring.get_config()

        assert isinstance(config, dict)
        assert 'units' in config
        assert config['units'] == xlstm_wiring.units


class TestAutoNCPWiring:
    """Test AutoNCP xLSTM wiring."""

    def test_create_autoncp_wiring(self):
        """Test creating AutoNCP wiring."""
        total_blocks = 32
        output_blocks = 4

        wiring = AutoNCPxLSTMWiring(
            total_blocks=total_blocks,
            output_blocks=output_blocks,
            sparsity_level=0.5
        )

        assert wiring.units == total_blocks
        assert wiring.num_layers == 3  # inter, command, motor
        assert wiring.output_dim == output_blocks

    def test_autoncp_layers(self):
        """Test AutoNCP layer structure."""
        wiring = AutoNCPxLSTMWiring(
            total_blocks=32,
            output_blocks=4,
            sparsity_level=0.5
        )

        # Check that we have 3 layers
        for layer_id in range(3):
            neurons = wiring.get_neurons_of_layer(layer_id)
            assert len(neurons) > 0

    def test_autoncp_neuron_types(self):
        """Test AutoNCP neuron types."""
        wiring = AutoNCPxLSTMWiring(
            total_blocks=32,
            output_blocks=4,
            sparsity_level=0.5
        )

        # Check neuron types match layers
        inter_neurons = wiring.get_neurons_of_layer(0)
        command_neurons = wiring.get_neurons_of_layer(1)
        motor_neurons = wiring.get_neurons_of_layer(2)

        for neuron_id in inter_neurons:
            assert wiring.get_type_of_neuron(neuron_id) == "inter"

        for neuron_id in command_neurons:
            assert wiring.get_type_of_neuron(neuron_id) == "command"

        for neuron_id in motor_neurons:
            assert wiring.get_type_of_neuron(neuron_id) == "motor"


class TestWiringConnectivity:
    """Test wiring connectivity patterns."""

    def test_add_synapse(self):
        """Test adding synapses to wiring."""
        wiring = xLSTMWiring(num_blocks=4)

        # Add excitatory synapse
        wiring.add_synapse(0, 1, polarity=1)
        assert wiring.adjacency_matrix[0][1] == 1

        # Add inhibitory synapse
        wiring.add_synapse(1, 2, polarity=-1)
        assert wiring.adjacency_matrix[1][2] == -1

    def test_sensory_synapse(self):
        """Test adding sensory synapses."""
        wiring = xLSTMWiring(num_blocks=4)
        wiring.build(input_dim=10)

        # Add sensory synapse from input to neuron
        wiring.add_sensory_synapse(0, 1, polarity=1)
        assert wiring.sensory_adjacency_matrix[0][1] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

