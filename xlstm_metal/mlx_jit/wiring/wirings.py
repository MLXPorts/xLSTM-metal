"""Pure-MLX wiring utilities for constructing sparse neural circuits."""

from typing import Dict, List, Optional

import mlx.core as mx


def _set_matrix_entry(matrix: mx.array, row: int, col: int, value: int) -> None:
    matrix[row, col] = mx.array(value, dtype=matrix.dtype)


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
        wiring = cls(config['units'])
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
        connections: List[str] = []
        for src in range(self.units):
            for dest in range(self.units):
                val = adj[src][dest]
                if val != 0:
                    polarity = "+" if val > 0 else "-"
                    connections.append(f"  neuron_{src} -> neuron_{dest} [{polarity}]")
        if connections:
            for line in connections:
                print(line)
        else:
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
