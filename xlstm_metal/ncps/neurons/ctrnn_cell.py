"""Back-compat import for the CTRNN-SE cell."""

from .ctrnn_se_cell import CTRNNSECell as CTRNNCell

__all__ = ["CTRNNCell", "CTRNNSECell"]
