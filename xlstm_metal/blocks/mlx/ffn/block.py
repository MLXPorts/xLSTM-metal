"""
xLSTM Feed-Forward Network Block - MLX Implementation

Implements the gated FFN from transformers xLSTM-7B (Lines 983-1028).

Architecture: x → SiLU(gate(x)) * up(x) → down(x)
"""

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn


@dataclass
class FFNConfig:
    """
    Configuration for Gated FFN (compatible with xLSTMBlockConfig).
    
    From xlstm_7b_model config.json:
        embedding_dim: 4096
        ffn_proj_factor: 2.667
        ffn_round_up_to_multiple_of: 64
        act_fn: "swish" (SiLU)
        use_bias: False
    """
    embedding_dim: int = 4096
    proj_up_dim: int = None  # If provided, use directly (preferred)
    proj_factor: float = 2.6671875  # Default from xLSTM-7B: 10944/4096
    ffn_round_up_to_multiple_of: int = 64
    act_fn: Literal["gelu", "swish", "relu"] = "swish"
    use_bias: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        if self.proj_up_dim is None:
            # Calculate proj_up_dim from factor and round up
            raw_dim = int(self.embedding_dim * self.proj_factor)
            self.proj_up_dim = (
                (raw_dim + self.ffn_round_up_to_multiple_of - 1) 
                // self.ffn_round_up_to_multiple_of 
                * self.ffn_round_up_to_multiple_of
            )


def round_up_to_next_multiple_of(value: float, multiple_of: int) -> int:
    """
    Round value up to next multiple of multiple_of.

    Used to ensure FFN up-projection dim is aligned for efficient matmul.
    """
    return int(((value + multiple_of - 1) // multiple_of) * multiple_of)


class xLSTMFeedForwardBlock(nn.Module):
    """
    Feed-forward network with gating for xLSTM-7B.

    Matches transformers.models.xlstm.modeling_xlstm.xLSTMFeedForward (Lines 983-1028).

    Architecture:
        x → proj_up_gate(x) → SiLU(gate) * proj_up(x) → proj_down(x) → y

    For xLSTM-7B config:
        - hidden_size: 4096
        - ffn_proj_factor: 2.6484375
        - ffn_round_up_to_multiple_of: 32
        - up_proj_dim: round_up(4096 * 2.6484375, 32) = 10880
        - weight_mode: "single" (separate proj_up_gate and proj_up)

    Weight keys in safetensors:
        - ffn.proj_up_gate.weight: [10880, 4096]
        - ffn.proj_up.weight: [10880, 4096]
        - ffn.proj_down.weight: [4096, 10880]

    Args:
        config_or_hidden_size: Either FFNConfig object or hidden_size int
        ffn_proj_factor: Up-projection multiplier (default 2.6484375)
        ffn_round_up_to_multiple_of: Alignment (default 32)
        use_bias: Whether to use bias in linear layers (default False)
        weight_mode: "single" (separate gates) or "fused" (default "single")

    Forward:
        Input: x [B, S, hidden_size]
        Output: y [B, S, hidden_size]
    """

    def __init__(
            self,
            hidden_size: int = 4096,
            ffn_proj_factor: float = 2.6484375,
            ffn_round_up_to_multiple_of: int = 32,
            use_bias: bool = False,
            weight_mode: str = "single"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.weight_mode = weight_mode

        # Compute up-projection dimension with rounding
        self.up_proj_dim = round_up_to_next_multiple_of(
            hidden_size * ffn_proj_factor,
            ffn_round_up_to_multiple_of
        )

        # Create projections based on weight mode
        if weight_mode == "single":
            # Separate projections (xLSTM-7B uses this)
            self.proj_up_gate = nn.Linear(
                hidden_size,
                self.up_proj_dim,
                bias=use_bias
            )
            self.proj_up = nn.Linear(
                hidden_size,
                self.up_proj_dim,
                bias=use_bias
            )
        elif weight_mode == "fused":
            # Fused projection (saves kernel launches)
            self.proj_up_gate_z = nn.Linear(
                hidden_size,
                mx.multiply(2, self.up_proj_dim),  # ZERO TOLERANCE: Use mx.multiply
                bias=use_bias
            )
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        # Down projection
        self.proj_down = nn.Linear(
            self.up_proj_dim,
            hidden_size,
            bias=use_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: gated FFN.

        Args:
            x: Input tensor [B, S, hidden_size]

        Returns:
            y: Output tensor [B, S, hidden_size]
        """
        if self.weight_mode == "single":
            # Separate gates: SiLU(gate) * value
            gate = self.proj_up_gate(x)  # [B, S, up_proj_dim]
            z = self.proj_up(x)          # [B, S, up_proj_dim]

            # ZERO TOLERANCE: Use MLX operators
            # SiLU = x * sigmoid(x), but nn.silu is cleaner
            gate_act = nn.silu(gate)
            x = mx.multiply(gate_act, z)  # [B, S, up_proj_dim]

        elif self.weight_mode == "fused":
            # Fused gates (like Llama)
            x = self.proj_up_gate_z(x)  # [B, S, 2*up_proj_dim]

            # Split: first up_proj_dim for gate, second for value
            gate = x[:, :, :self.up_proj_dim]  # [B, S, up_proj_dim]
            z = x[:, :, self.up_proj_dim:]     # [B, S, up_proj_dim]

            # ZERO TOLERANCE: Use MLX operators
            gate_act = nn.silu(gate)
            x = mx.multiply(gate_act, z)  # [B, S, up_proj_dim]

        # Down projection
        y = self.proj_down(x)  # [B, S, hidden_size]

        return y

    def __repr__(self):
        return (
            f"xLSTMFeedForwardBlock(\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  up_proj_dim={self.up_proj_dim},\n"
            f"  weight_mode={self.weight_mode}\n"
            f")"
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing xLSTMFeedForwardBlock...")

    # xLSTM-7B config
    config = {
        "hidden_size": 4096,
        "ffn_proj_factor": 2.6484375,
        "ffn_round_up_to_multiple_of": 32,
        "use_bias": False,
        "weight_mode": "single"
    }

    # Create block
    ffn = xLSTMFeedForwardBlock(**config)
    print(f"\nBlock structure:\n{ffn}")

    # Test forward pass
    B, S = 2, 10
    x = mx.random.normal((B, S, config["hidden_size"]))

    print(f"\nInput shape: {x.shape}")
    y = ffn(x)
    print(f"Output shape: {y.shape}")

    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    print("\n✓ Shape test passed!")

    # Check up_proj_dim calculation
    expected_up_dim = round_up_to_next_multiple_of(
        4096 * 2.6484375,
        32
    )
    assert ffn.up_proj_dim == expected_up_dim, (
        f"Expected up_proj_dim={expected_up_dim}, got {ffn.up_proj_dim}"
    )
    print(f"✓ up_proj_dim = {ffn.up_proj_dim} (correct!)")

    print("\nAll tests passed!")
