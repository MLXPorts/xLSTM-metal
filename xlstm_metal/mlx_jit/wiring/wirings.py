"""Pure-MLX wiring utilities for constructing sparse neural circuits."""

from random import Random as PyRandom
from typing import Dict, Iterable, List, Optional

import mlx.core as mx


def _set_matrix_entry(matrix: mx.array, row: int, col: int, value: int) -> None:
    matrix[row, col] = int(value)


class Wiring:
    """Connectivity blueprint describing synapses between neurons."""

    def __init__(self, units: int) -> None:
        self.units = units
        self.adjacency_matrix = mx.zeros((units, units), dtype=mx.int32)
        self.sensory_adjacency_matrix: Optional[mx.array] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    # ------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        """

        :return:
        """
        return 1

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """

        :param layer_id:
        :return:
        """
        return list(range(self.units))

    def is_built(self) -> bool:
        """

        :return:
        """
        return self.input_dim is not None

    def build(self, input_dim: int) -> None:
        """

        :param input_dim:
        """
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided: expected "
                f"{self.input_dim}, got {input_dim}."
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    # ------------------------------------------------------------------
    def erev_initializer(self, *_: object, **__: object) -> mx.array:
        """

        :param _:
        :param __:
        :return:
        """
        return mx.array(self.adjacency_matrix, dtype=mx.int32)

    def sensory_erev_initializer(self, *_: object, **__: object) -> mx.array:
        """

        :param _:
        :param __:
        :return:
        """
        if self.sensory_adjacency_matrix is None:
            raise ValueError("Sensory adjacency matrix not initialised.")
        return mx.array(self.sensory_adjacency_matrix, dtype=mx.int32)

    def set_input_dim(self, input_dim: int) -> None:
        """

        :param input_dim:
        """
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = mx.zeros(
            (input_dim, self.units), dtype=mx.int32
        )

    def set_output_dim(self, output_dim: int) -> None:
        """

        :param output_dim:
        """
        self.output_dim = output_dim

    # ------------------------------------------------------------------
    def get_type_of_neuron(self, neuron_id: int) -> str:
        """

        :param neuron_id:
        :return:
        """
        if self.output_dim is None:
            return "inter"
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src: int, dest: int, polarity: int) -> None:
        """

        :param src:
        :param dest:
        :param polarity:
        """
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid source neuron {src} for {self.units} units")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        _set_matrix_entry(self.adjacency_matrix, src, dest, polarity)

    def add_sensory_synapse(self, src: int, dest: int, polarity: int) -> None:
        """

        :param src:
        :param dest:
        :param polarity:
        """
        if self.input_dim is None or self.sensory_adjacency_matrix is None:
            raise ValueError("Cannot add sensory synapse before build().")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"Invalid sensory index {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        _set_matrix_entry(self.sensory_adjacency_matrix, src, dest, polarity)

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, object]:
        """

        :return:
        """
        return {
            "units": self.units,
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "sensory_adjacency_matrix": None
            if self.sensory_adjacency_matrix is None
            else self.sensory_adjacency_matrix.tolist(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "Wiring":
        """

        :param config:
        :return:
        """
        wiring = cls(int(config['units']))
        wiring.adjacency_matrix = mx.array(
            config["adjacency_matrix"], dtype=mx.int32
        )
        if config.get("sensory_adjacency_matrix") is not None:
            wiring.sensory_adjacency_matrix = mx.array(
                config["sensory_adjacency_matrix"], dtype=mx.int32
            )
        wiring.input_dim = config.get("input_dim")
        wiring.output_dim = config.get("output_dim")
        return wiring

    # ------------------------------------------------------------------
    def get_graph(self, include_sensory_neurons: bool = True):  # pragma: no cover
        """

        :param include_sensory_neurons:
        :return:
        """
        if not self.is_built():
            raise ValueError(
                "Wiring is not built yet; call build() with the input dimension."
            )
        import networkx as nx

        graph = nx.DiGraph()
        for i in range(self.units):
            graph.add_node(
                f"neuron_{i}", neuron_type=self.get_type_of_neuron(i)
            )
        if include_sensory_neurons and self.input_dim is not None:
            for i in range(self.input_dim):
                graph.add_node(f"sensory_{i}", neuron_type="sensory")

        adj = self.adjacency_matrix.tolist()
        sensory_adj = (
            self.sensory_adjacency_matrix.tolist()
            if self.sensory_adjacency_matrix is not None
            else []
        )

        if include_sensory_neurons and sensory_adj:
            for src in range(self.input_dim):
                for dest in range(self.units):
                    if sensory_adj[src][dest] != 0:
                        polarity = (
                            "excitatory"
                            if sensory_adj[src][dest] >= 0
                            else "inhibitory"
                        )
                        graph.add_edge(
                            f"sensory_{src}", f"neuron_{dest}", polarity=polarity
                        )

        for src in range(self.units):
            for dest in range(self.units):
                if adj[src][dest] != 0:
                    polarity = "excitatory" if adj[src][dest] >= 0 else "inhibitory"
                    graph.add_edge(
                        f"neuron_{src}", f"neuron_{dest}", polarity=polarity
                    )
        return graph

    def draw_graph(  # pragma: no cover
            self,
            layout: str = "shell",
            neuron_colors: Optional[Dict[str, str]] = None,
            synapse_colors: Optional[Dict[str, str]] = None,
            draw_labels: bool = False,
    ):
        """

        :param layout:
        :param neuron_colors:
        :param synapse_colors:
        :param draw_labels:
        :return:
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import networkx as nx

        if synapse_colors is None:
            synapse_colors = {"excitatory": "tab:green", "inhibitory": "tab:red"}
        elif isinstance(synapse_colors, str):
            synapse_colors = {
                "excitatory": synapse_colors,
                "inhibitory": synapse_colors,
            }

        default_colors = {
            "inter": "tab:blue",
            "motor": "tab:orange",
            "sensory": "tab:olive",
        }
        palette = dict(default_colors)
        if neuron_colors:
            palette.update(neuron_colors)

        legend_patches = [
            mpatches.Patch(color=color, label=f"{name.title()} neurons")
            for name, color in palette.items()
        ]

        graph = self.get_graph()
        layouts = {
            "kamada": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }
        if layout not in layouts:
            raise ValueError(f"Unknown layout '{layout}'")
        pos = layouts[layout](graph)

        for node, data in graph.nodes(data=True):
            color = palette.get(data.get("neuron_type", "inter"), "tab:blue")
            nx.draw_networkx_nodes(graph, pos, [node], node_color=color)

        if draw_labels:
            nx.draw_networkx_labels(graph, pos)

        for node1, node2, data in graph.edges(data=True):
            edge_color = synapse_colors[data["polarity"]]
            nx.draw_networkx_edges(graph, pos, [(node1, node2)], edge_color=edge_color)

        plt.axis("off")
        return legend_patches

    def print_diagram(self, include_sensory: bool = True) -> None:
        """Print a simple textual wiring diagram (src -> dest [polarity])."""
        print("\nWIRING DIAGRAM")
        print("--------------")

        if include_sensory and self.sensory_adjacency_matrix is not None and self.input_dim:
            print("Sensory connections:")
            sensory = self.sensory_adjacency_matrix.tolist()
            for src in range(self.input_dim):
                for dest in range(self.units):
                    val = sensory[src][dest]
                    if val != 0:
                        polarity = "+" if val > 0 else "-"
                        print(f"  sensory_{src} -> neuron_{dest} [{polarity}]")

        print("Neural connections:")
        adj = self.adjacency_matrix.tolist()
        found = False
        for src in range(self.units):
            for dest in range(self.units):
                val = adj[src][dest]
                if val != 0:
                    polarity = "+" if val > 0 else "-"
                    print(f"  neuron_{src} -> neuron_{dest} [{polarity}]")
                    found = True
        if not found:
            print("  (no synapses)")

    # ------------------------------------------------------------------
    @property
    def synapse_count(self) -> mx.array:
        """

        :return:
        """
        return mx.sum(mx.abs(self.adjacency_matrix))

    @property
    def sensory_synapse_count(self) -> mx.array:
        """

        :return:
        """
        if self.sensory_adjacency_matrix is None:
            return mx.array(0, dtype=mx.int32)
        return mx.sum(mx.abs(self.sensory_adjacency_matrix))


class FullyConnected(Wiring):
    def __init__(
            self, units: int, output_dim: Optional[int] = None, erev_init_seed: int = 1111,
            self_connections: bool = True
    ) -> None:
        super().__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = PyRandom(erev_init_seed)
        self._seed = erev_init_seed

        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                self.add_synapse(src, dest, self._polarity())

    def _polarity(self) -> int:
        return -1 if self._rng.random() < (1.0 / 3.0) else 1

    def build(self, input_shape: int) -> None:
        """

        :param input_shape:
        """
        super().build(input_shape)
        assert self.input_dim is not None
        for src in range(self.input_dim):
            for dest in range(self.units):
                self.add_sensory_synapse(src, dest, self._polarity())

    def get_config(self) -> Dict[str, object]:
        """

        :return:
        """
        return {
            "units": self.units,
            "output_dim": self.output_dim,
            "erev_init_seed": self._seed,
            "self_connections": self.self_connections,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "FullyConnected":
        """

        :param config:
        :return:
        """
        return cls(**config)


class Random(Wiring):
    def __init__(
            self,
            units: int,
            output_dim: Optional[int] = None,
            sparsity_level: float = 0.0,
            random_seed: int = 1111,
    ) -> None:
        super().__init__(units)
        if output_dim is None:
            output_dim = units
        if sparsity_level < 0.0 or sparsity_level >= 1.0:
            raise ValueError("sparsity_level must be in [0, 1)")
        self.set_output_dim(output_dim)
        self.sparsity_level = sparsity_level
        self._rng = PyRandom(random_seed)
        self._seed = random_seed

        total = self.units * self.units
        synapses = round(total * (1.0 - sparsity_level))
        all_synapses = [(src, dest) for src in range(self.units) for dest in range(self.units)]
        for src, dest in self._rng.sample(all_synapses, synapses):
            self.add_synapse(src, dest, self._polarity())

    def _polarity(self) -> int:
        return -1 if self._rng.random() < (1.0 / 3.0) else 1

    def build(self, input_shape: int) -> None:
        """

        :param input_shape:
        """
        super().build(input_shape)
        assert self.input_dim is not None
        total = self.input_dim * self.units
        synapses = round(total * (1.0 - self.sparsity_level))
        all_synapses = [(src, dest) for src in range(self.input_dim) for dest in range(self.units)]
        for src, dest in self._rng.sample(all_synapses, synapses):
            self.add_sensory_synapse(src, dest, self._polarity())

    def get_config(self) -> Dict[str, object]:
        """

        :return:
        """
        return {
            "units": self.units,
            "output_dim": self.output_dim,
            "sparsity_level": self.sparsity_level,
            "random_seed": self._seed,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "Random":
        """

        :param config:
        :return:
        """
        return cls(**config)


class NCP(Wiring):
    def __init__(
            self,
            inter_neurons: int,
            command_neurons: int,
            motor_neurons: int,
            sensory_fanout: int,
            inter_fanout: int,
            recurrent_command_synapses: int,
            motor_fanin: int,
            seed: int = 22222,
    ) -> None:
        super().__init__(inter_neurons + command_neurons + motor_neurons)
        self._sensory_neurons = None
        self._num_sensory_neurons = None
        self.set_output_dim(motor_neurons)
        self._rng = PyRandom(seed)
        self._seed = seed

        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

        self._motor_neurons = list(range(0, motor_neurons))
        self._command_neurons = list(range(motor_neurons, motor_neurons + command_neurons))
        self._inter_neurons = list(
            range(
                motor_neurons + command_neurons,
                motor_neurons + command_neurons + inter_neurons,
            )
        )

        if motor_fanin > command_neurons:
            raise ValueError("motor_fanin exceeds number of command neurons")
        if sensory_fanout > inter_neurons:
            raise ValueError("sensory_fanout exceeds number of inter neurons")
        if inter_fanout > command_neurons:
            raise ValueError("inter_fanout exceeds number of command neurons")

    @property
    def num_layers(self) -> int:
        """

        :return:
        """
        return 3

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """

        :param layer_id:
        :return:
        """
        if layer_id == 0:
            return list(self._inter_neurons)
        if layer_id == 1:
            return list(self._command_neurons)
        if layer_id == 2:
            return list(self._motor_neurons)
        raise ValueError(f"Unknown layer {layer_id}")

    def get_type_of_neuron(self, neuron_id: int) -> str:
        """

        :param neuron_id:
        :return:
        """
        if neuron_id in self._motor_neurons:
            return "motor"
        if neuron_id in self._command_neurons:
            return "command"
        return "inter"

    def _polarity(self) -> int:
        return -1 if self._rng.random() < 0.5 else 1

    def _choose(self, seq: Iterable[int]) -> int:
        return self._rng.sample(list(seq), 1)[0]

    def _build_sensory_to_inter_layer(self) -> None:
        assert self.input_dim is not None and self.sensory_adjacency_matrix is not None
        unreachable = set(self._inter_neurons)
        for src in range(self.input_dim):
            dests = self._rng.sample(self._inter_neurons, min(self._sensory_fanout, len(self._inter_neurons)))
            for dest in dests:
                unreachable.discard(dest)
                self.add_sensory_synapse(src, dest, self._polarity())

        if unreachable:
            fanin = max(1, min(self.input_dim, round(self.input_dim * self._sensory_fanout / self._num_inter_neurons)))
            for dest in unreachable:
                for src in self._rng.sample(list(range(self.input_dim)), fanin):
                    self.add_sensory_synapse(src, dest, self._polarity())

    def _build_inter_to_command_layer(self) -> None:
        unreachable = set(self._command_neurons)
        for src in self._inter_neurons:
            dests = self._rng.sample(self._command_neurons, min(self._inter_fanout, len(self._command_neurons)))
            for dest in dests:
                unreachable.discard(dest)
                self.add_synapse(src, dest, self._polarity())

        if unreachable:
            fanin = max(1, min(self._num_inter_neurons,
                               round(self._num_inter_neurons * self._inter_fanout / self._num_command_neurons)))
            for dest in unreachable:
                for src in self._rng.sample(self._inter_neurons, fanin):
                    self.add_synapse(src, dest, self._polarity())

    def _build_recurrent_command_layer(self) -> None:
        for _ in range(self._recurrent_command_synapses):
            src = self._choose(self._command_neurons)
            dest = self._choose(self._command_neurons)
            self.add_synapse(src, dest, self._polarity())

    def _build_command_to_motor_layer(self) -> None:
        unreachable = set(self._command_neurons)
        for dest in self._motor_neurons:
            fanin = min(self._motor_fanin, len(self._command_neurons))
            srcs = self._rng.sample(self._command_neurons, fanin)
            for src in srcs:
                unreachable.discard(src)
                self.add_synapse(src, dest, self._polarity())

        if unreachable:
            fanout = max(1, min(self._num_motor_neurons,
                                round(self._num_motor_neurons * self._motor_fanin / self._num_command_neurons)))
            for src in unreachable:
                for dest in self._rng.sample(self._motor_neurons, fanout):
                    self.add_synapse(src, dest, self._polarity())

    def build(self, input_shape: int) -> None:
        """

        :param input_shape:
        """
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(self._num_sensory_neurons))
        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command_to_motor_layer()

    def get_config(self) -> Dict[str, object]:
        """

        :return:
        """
        return {
            "inter_neurons": self._num_inter_neurons,
            "command_neurons": self._num_command_neurons,
            "motor_neurons": self._num_motor_neurons,
            "sensory_fanout": self._sensory_fanout,
            "inter_fanout": self._inter_fanout,
            "recurrent_command_synapses": self._recurrent_command_synapses,
            "motor_fanin": self._motor_fanin,
            "seed": self._seed,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "NCP":
        """

        :param config:
        :return:
        """
        return cls(**config)


class AutoNCP(NCP):
    def __init__(
            self,
            units: int,
            output_size: int,
            sparsity_level: float = 0.5,
            seed: int = 22222,
    ) -> None:
        if output_size >= units - 2:
            raise ValueError(
                f"Output size must be less than units-2 (given {units=} and {output_size=})."
            )
        if not (0.0 < sparsity_level <= 1.0):
            raise ValueError("sparsity_level must be in (0, 1]")

        density = 1.0 - sparsity_level
        inter_and_command = units - output_size
        command = max(int(0.4 * inter_and_command), 1)
        inter = inter_and_command - command

        sensory_fanout = max(int(inter * density), 1)
        inter_fanout = max(int(command * density), 1)
        recurrent_command = max(int(command * density * 2), 1)
        motor_fanin = max(int(command * density), 1)

        super().__init__(
            inter,
            command,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command,
            motor_fanin,
            seed=seed,
        )
        self._total_units = units
        self._output_size = output_size
        self._sparsity_level = sparsity_level

    def get_config(self) -> Dict[str, object]:
        """

        :return:
        """
        return {
            "units": self._total_units,
            "output_size": self._output_size,
            "sparsity_level": self._sparsity_level,
            "seed": self._seed,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "AutoNCP":
        """

        :param config:
        :return:
        """
        return cls(**config)
