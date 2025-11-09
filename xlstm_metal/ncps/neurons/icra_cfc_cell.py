"""CfC cell tailored to the ICRA lidar regression task.

This module re-implements, from scratch, the conv feature head and data
fusion pattern used in the original ICRA LDS repository (CTRNN/LSTM
baselines). The architecture mirrors:

  - Conv1D(12, k=5, s=3) → Conv1D(16, k=5, s=3) → Conv1D(24, k=5, s=2)
    → Conv1D(1, k=1, s=1), followed by flatten-per-timestep and concatenation
    with two clipped vehicle-state scalars.

No code has been copied from the upstream repository; this MLX version follows
the documented/observable behaviour and uses our profile-backed CfC core.
"""

from __future__ import annotations

from typing import Optional, Union, Dict

import mlx.core as mx # noqa
import mlx.nn as nn # noqa
from .hyperprofiles import HyperProfile
from .cfc_profiled import CfCProfiled


class IcraCfCCell(nn.Module):
    """Wraps a profile-backed CfC with the lidar Conv1D feature extractor."""

    def __init__(
        self,
        lidar_bins: int,
        state_dim: int = 2,
        profile: Union[str, HyperProfile] = "cfc_icra",
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.lidar_bins = int(lidar_bins)
        self.state_dim = int(state_dim)
        self.profile_name = profile

        # MLX-idiomatic conv head (channel-last: [N, L, C])
        self.head = nn.Sequential(nn.Conv1d(1, 12, kernel_size=5, stride=3), nn.ReLU(),
            nn.Conv1d(12, 16, kernel_size=5, stride=3), nn.ReLU(),
            nn.Conv1d(16, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv1d(24, 1, kernel_size=1, stride=1),
        )

        # Derive feature dimension by probing once (no manual formulas)
        probe = mx.zeros((1, self.lidar_bins, 1), dtype=mx.float32)
        feature_dim = self.head(probe).shape[1]
        self.core = CfCProfiled(
            input_size=mx.add(feature_dim, self.state_dim),
            profile=profile,
            overrides=overrides,
        )

    def _encode_lidar(self, lidar: mx.array) -> mx.array:
        batch, steps, bins = lidar.shape
        x = mx.reshape(lidar, (mx.multiply(batch, steps), bins, 1))
        x = self.head(x)
        x = mx.reshape(x, (batch, steps, -1))
        return x

    def __call__(
        self,
        state_seq: mx.array,
        lidar_seq: mx.array,
        *,
        hx: Optional[mx.array] = None,
        return_state: bool = False,
    ):
        if lidar_seq.shape[-1] != self.lidar_bins:
            raise ValueError("Unexpected lidar dimension")
        if state_seq.shape[-1] != self.state_dim:
            raise ValueError("Unexpected state dimension")

        features = self._encode_lidar(lidar_seq)
        state_clipped = mx.clip(state_seq, -1.0, 1.0)
        inputs = mx.concatenate([features, state_clipped], axis=-1)
        return self.core(inputs, hx=hx, return_state=return_state)

    def apply_constraints(self) -> None:
        self.core.apply_constraints()

    def to_config(self) -> dict:
        return {
            "cell": "IcraCfCCell",
            "lidar_bins": self.lidar_bins,
            "state_dim": self.state_dim,
            "profile": getattr(self.core, "profile", None).name if hasattr(self.core, "profile") else self.profile_name,
            "profile_extras": getattr(self.core, "profile", None).extras if hasattr(self.core, "profile") else {},
            "cfc_config": getattr(self.core, "config", {}),
        }

    # ---- Inference helpers -------------------------------------------------
    @property
    def state_size(self) -> int:
        return int(self.core.model.state_size)

    def zero_state(self, batch_size: int = 1) -> mx.array:
        return mx.zeros((batch_size, self.state_size), dtype=mx.float32)

    def step(
        self,
        state_t: mx.array,
        lidar_t: mx.array,
        *,
        hx: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Single-timestep inference.

        Args:
            state_t: [B, 2] vehicle state.
            lidar_t: [B, L] lidar bins.
            hx: optional hidden state [B, H].

        Returns:
            (y_t, hx_next) where y_t is [B, 1].
        """
        if state_t.ndim == 1:
            state_t = mx.expand_dims(state_t, axis=0)
        if lidar_t.ndim == 1:
            lidar_t = mx.expand_dims(lidar_t, axis=0)
        state_seq = mx.expand_dims(state_t, axis=1)  # [B,1,2]
        lidar_seq = mx.expand_dims(lidar_t, axis=1)  # [B,1,L]
        y_seq, hx_next = self(state_seq, lidar_seq, hx=hx, return_state=True)
        return y_seq[:, 0, :], hx_next
