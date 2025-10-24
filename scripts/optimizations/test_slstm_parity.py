import os
import torch

from xlstm_official_full.blocks.slstm.cell import sLSTMCell, sLSTMCellConfig


def main():
    assert torch.backends.mps.is_available(), "MPS not available; run on Apple Silicon with MPS."
    device = torch.device("mps")

    torch.manual_seed(0)

    B, S = 2, 5
    hidden_size = 64
    num_heads = 4
    num_gates = 4

    # Build vanilla and compiled cells with identical configs
    cfg_base = sLSTMCellConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dtype="float32",
        enable_automatic_mixed_precision=False,
    )

    cfg_vanilla = sLSTMCellConfig(**vars(cfg_base))
    cfg_vanilla.backend = "vanilla"
    cell_vanilla = sLSTMCell(cfg_vanilla, skip_backend_init=True).to(device).eval()

    cfg_compiled = sLSTMCellConfig(**vars(cfg_base))
    cfg_compiled.backend = "native_compiled"
    cell_compiled = sLSTMCell(cfg_compiled, skip_backend_init=True).to(device).eval()

    x = torch.randn(B, S, num_gates * hidden_size, device=device, dtype=torch.float32)

    with torch.inference_mode():
        y_v, s_v = cell_vanilla(x)
        y_c, s_c = cell_compiled(x)

    y_err = (y_v - y_c).abs().max().item()
    s_err = (s_v - s_c).abs().max().item()

    print(f"sLSTM parity max-abs error: y={y_err:.3e}, state={s_err:.3e}")
    # Fail loudly if not numerically identical within tight tolerance
    assert y_err < 1e-6 and s_err < 1e-6, "sLSTM compiled parity check failed"


if __name__ == "__main__":
    main()

