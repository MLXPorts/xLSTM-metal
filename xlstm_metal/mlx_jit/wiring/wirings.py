"""NCPS Wiring – MLX Implementation (Sparse Neural Circuit Blueprints)

Overview
--------
Wiring defines the **connectivity blueprint** for sparse neural circuits in
the NCPS (Neural Circuit Policies) framework. Instead of densely connecting
all neurons, wiring specifies which neurons connect to which (analogous to
biological synaptic connectivity patterns).

NCPS Philosophy
---------------
Traditional neural networks use dense weight matrices where every input
connects to every output. NCPS instead defines:
  - **Neurons**: Computational units (e.g., LSTM cells, attention heads)
  - **Synapses**: Directed connections between neurons with polarity
  - **Wiring**: The adjacency matrix encoding which synapses exist

This modularity enables:
  1. Compositional architectures (mix different cell types)
  2. Sparse connectivity (reduce parameters, improve interpretability)
  3. Biologically-inspired circuit motifs (feed-forward, recurrent, lateral inhibition)

Neuron Types
------------
- **Sensory**: Input neurons receiving external features
- **Inter**: Internal/hidden neurons (computation, memory)
- **Motor**: Output neurons producing predictions/actions

Synapse Polarity
---------------
Each synapse has polarity ∈ {-1, +1}:
  - **Excitatory (+1)**: Positive influence (strengthen activation)
  - **Inhibitory (-1)**: Negative influence (suppress activation)

In xLSTM context, polarity is typically +1 (standard feed-forward flow).
Inhibitory synapses can model gating or competition between pathways.

Adjacency Matrices
------------------
Two connectivity matrices:
  1. **adjacency_matrix**: [units, units] inter-neuron connections
  2. **sensory_adjacency_matrix**: [input_dim, units] sensory → inter connections

Entry values:
  - 0: No synapse
  - +1: Excitatory synapse
  - -1: Inhibitory synapse

Sequential Wiring Pattern (xLSTM)
---------------------------------
For transformer/LSTM-style models, wiring is typically **sequential**:
  block_0 → block_1 → block_2 → ... → block_N → output

Each block is a "neuron" in NCPS terminology, and sequential connectivity
ensures information flows through the entire stack.

Usage in xLSTM
--------------
While xLSTM models use sequential stacking (not sparse wiring), the Wiring
abstraction provides:
  - Uniform interface for model assembly
  - Introspection (query block types, connectivity)
  - Extensibility (future sparse/mixture variants)

Building Wiring
---------------
1. Create wiring: `wiring = Wiring(units=32)`
2. Add synapses: `wiring.add_synapse(src=0, dest=1, polarity=1)`
3. Set input: `wiring.build(input_dim=256)`
4. Add sensory: `wiring.add_sensory_synapse(src=0, dest=5, polarity=1)`

Serialization
-------------
Wiring can be saved/loaded via `get_config()` / `from_config()` for
reproducibility and transfer across frameworks.

Visualization
-------------
The `draw_graph()` method renders the wiring as a NetworkX graph for
inspection and debugging.

Parity
------
Logic mirrors torch-native Wiring for cross-backend compatibility.
"""

from typing import Dict, List, Optional

import mlx.core as mx


def _set_matrix_entry(matrix: mx.array, row: int, col: int, value: int) -> None:
    matrix[row, col] = mx.array(value, dtype=matrix.dtype)


class Wiring:
    """Connectivity blueprint for sparse neural circuits (NCPS framework).

    Encodes which neurons connect to which via adjacency matrices. Supports
    both inter-neuron (recurrent) and sensory (input) connections. Provides
    serialization, visualization, and introspection methods.

    Parameters
    ----------
    units : int
        Number of neurons in the circuit (excluding sensory inputs).

    Attributes
    ----------
    adjacency_matrix : mx.array [units, units]
        Inter-neuron connectivity matrix (0 = no synapse, ±1 = synapse polarity).
    sensory_adjacency_matrix : mx.array [input_dim, units] | None
        Sensory → inter-neuron connectivity (set after build()).
    input_dim : int | None
        Number of sensory input features (set via build()).
    output_dim : int | None
        Number of motor output neurons (set via set_output_dim()).

    Methods
    -------
    build(input_dim)
        Initialize sensory connectivity matrix.
    add_synapse(src, dest, polarity)
        Add inter-neuron synapse.
    add_sensory_synapse(src, dest, polarity)
        Add sensory → inter-neuron synapse.
    get_config() / from_config(config)
        Serialize/deserialize wiring for persistence.
    draw_graph(layout, colors, labels)
        Visualize wiring as NetworkX graph (requires matplotlib).
    """

    def __init__(self, units: int) -> None:
        self.units = units
        self.adjacency_matrix = mx.zeros((units, units))
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
        """Initialize wiring with input dimension (creates sensory adjacency).

        Parameters
        ----------
        input_dim : int
            Number of sensory input features.

        Raises
        ------
        ValueError
            If input_dim conflicts with previously set value.
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
        return mx.array(self.adjacency_matrix)

    def sensory_erev_initializer(self, *_: object, **__: object) -> mx.array:
        """

        :param _:
        :param __:
        :return:
        """
        if self.sensory_adjacency_matrix is None:
            raise ValueError("Sensory adjacency matrix not initialised.")
        return mx.array(self.sensory_adjacency_matrix)

    def set_input_dim(self, input_dim: int) -> None:
        """

        :param input_dim:
        """
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = mx.zeros(
            (input_dim, self.units)
        )

    def set_output_dim(self, output_dim: int) -> None:
        """

        :param output_dim:
        """
        self.output_dim = output_dim

    # ------------------------------------------------------------------
    def get_type_of_neuron(self, neuron_id: int) -> str:
        """Classify neuron as 'motor' (output) or 'inter' (hidden).

        Parameters
        ----------
        neuron_id : int
            Neuron index.

        Returns
        -------
        neuron_type : {"motor", "inter"}
            Neuron classification based on output_dim.
        """
        if self.output_dim is None:
            return "inter"
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src: int, dest: int, polarity: int = 1) -> None:
        """Add inter-neuron synapse (recurrent connection).

        Parameters
        ----------
        src : int
            Source neuron index [0, units).
        dest : int
            Destination neuron index [0, units).
        polarity : {-1, +1}, default 1
            Synapse polarity (excitatory +1, inhibitory -1).

        Raises
        ------
        ValueError
            If indices out of bounds or invalid polarity.
        """
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid source neuron {src} for {self.units} units")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in (-1, 1):
            raise ValueError("Synapse polarity must be -1 or +1")
        _set_matrix_entry(self.adjacency_matrix, src, dest, polarity)

    def add_sensory_synapse(self, src: int, dest: int, polarity: int = 1) -> None:
        """Add sensory → inter-neuron synapse (input connection).

        Parameters
        ----------
        src : int
            Sensory input index [0, input_dim).
        dest : int
            Destination neuron index [0, units).
        polarity : {-1, +1}, default 1
            Synapse polarity.

        Raises
        ------
        ValueError
            If wiring not built or indices out of bounds.
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
            config["adjacency_matrix"]
        )
        if config.get("sensory_adjacency_matrix") is not None:
            wiring.sensory_adjacency_matrix = mx.array(
                config["sensory_adjacency_matrix"]
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

    def draw_graph(
            self,
            layout: str = "shell",
            neuron_colors: Optional[Dict[str, str]] = None,
            synapse_colors: Optional[Dict[str, str]] = None,
            draw_labels: bool = False,
    ):  # pragma: no cover
        """Render wiring as NetworkX graph (requires matplotlib).

        Parameters
        ----------
        layout : str, default "shell"
            NetworkX layout algorithm: {"kamada", "circular", "spring", "spectral", ...}
        neuron_colors : dict | None
            Mapping {neuron_type: color} for node coloring.
        synapse_colors : dict | None
            Mapping {polarity: color} for edge coloring.
        draw_labels : bool, default False
            Whether to annotate nodes with neuron IDs.

        Returns
        -------
        legend_patches : list
            Matplotlib patch objects for legend creation.
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

    def print_diagram(self, include_sensory: bool = True, style: str = "unicode") -> None:
        """Print a textual wiring diagram with simple ASCII/Unicode glyphs."""

        style = style.lower()
        if style not in {"unicode", "ascii"}:
            raise ValueError("style must be 'unicode' or 'ascii'")

        arrow_exc = " ──▶ " if style == "unicode" else " --> "
        arrow_inh = " ──┤ " if style == "unicode" else " -x> "
        bullet = "•" if style == "unicode" else "*"
        header_top = "┏━━━━━━━━━━━━━━━━━━━━━━┓" if style == "unicode" else "===================="
        header_mid = "┃ WIRING DIAGRAM      ┃" if style == "unicode" else "== WIRING DIAGRAM =="
        header_bot = "┗━━━━━━━━━━━━━━━━━━━━━━┛" if style == "unicode" else "===================="

        def format_line(src_label: str, dest_label: str, polarity: int) -> str:
            arrow = arrow_exc if polarity > 0 else arrow_inh
            tag = "exc" if polarity > 0 else "inh"
            return f"  {bullet} {src_label}{arrow}{dest_label} ({tag})"

        print("\n" + header_top)
        print(header_mid)
        print(header_bot)

        if include_sensory and self.sensory_adjacency_matrix is not None and self.input_dim:
            print("Sensory → Neuron:")
            sensory = self.sensory_adjacency_matrix.tolist()
            for src in range(self.input_dim):
                for dest in range(self.units):
                    val = sensory[src][dest]
                    if val != 0:
                        print(format_line(f"sensory_{src:02d}", f"neuron_{dest:02d}", val))

        print("Neuron → Neuron:")
        adj = self.adjacency_matrix.tolist()
        connections: List[str] = []
        for src in range(self.units):
            for dest in range(self.units):
                val = adj[src][dest]
                if val != 0:
                    connections.append(
                        format_line(f"neuron_{src:02d}", f"neuron_{dest:02d}", val)
                    )
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
            return mx.array(0)
        return mx.sum(mx.abs(self.sensory_adjacency_matrix))


__all__ = ['Wiring']
