# Full API Reference (Draft)

This is a consolidated API reference for the reference modules.

---

## lnn_hrm.memory_cube

### class MemoryCube
- __init__(d_key: int, d_val: int, max_items: int = 65536, topk: int = 8, device: str = "cpu")
- query(q: Tensor) -> Tuple[Tensor, Tensor]
- update(k_new: Tensor, v_new: Tensor) -> None

Attributes:
- keys: [N, d_key]
- vals: [N, d_val]

---

## lnn_hrm.liquid_time_constant

### class LiquidTimeConstant
- __init__(input_size: int, hidden_size: int, tau_init: float = 1.0)
- forward(x: Tensor, h: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]

---

## lnn_hrm.cube_gated_block

### class CubeGatedBlock
- __init__(d_in: int, d_key: int = None, d_val: int = None)
- forward(h_in: Tensor, y_teacher: Optional[Tensor] = None, train: bool = False) -> Tuple[Tensor, float, float]

---

## lnn_hrm.transformer_lnn

### class TransformerLNN
- __init__(input_size: int, hidden_size: int, num_heads: int = 4, dropout: float = 0.1)
- forward(x: Tensor, times: Optional[Tensor] = None) -> Tuple[Tensor, Dict]

---

## examples.transformer_lnn_example

- main() entrypoint

