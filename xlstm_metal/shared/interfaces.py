"""Backend-agnostic protocols for cells, blocks, and wiring."""

from typing import Protocol, Any, Tuple


class Cell(Protocol):
    """Backend-agnostic cell interface."""

    def __call__(self, x: Any, state: Any | None = None) -> Tuple[Any, Any]:
        """
        Forward pass through the cell.

        Args:
            x: Input tensor
            state: Optional recurrent state

        Returns:
            Tuple of (output, new_state)
        """
        ...


class Wiring(Protocol):
    """Backend-agnostic wiring interface."""

    units: int
    output_dim: int | None
    input_dim: int | None

    def build(self, input_dim: int) -> None:
        """Build wiring with input dimensions."""
        ...

    def add_synapse(self, source: int, target: int, polarity: float) -> None:
        """Add a synapse between neurons."""
        ...

    def is_built(self) -> bool:
        """Check if wiring is built."""
        ...

