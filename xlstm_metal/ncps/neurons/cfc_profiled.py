"""Profile-driven CfC wrappers."""

from __future__ import annotations

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .cfc import CfC
from .hyperprofiles import HyperProfile, load_profile


class CfCProfiled(nn.Module):
    """Instantiate a CfC model using a saved hyperparameter profile."""

    def __init__(
        self,
        input_size: int,
        profile: str | HyperProfile,
        *,
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        profile_obj = load_profile(profile) if isinstance(profile, str) else profile
        config = dict(profile_obj.extras)
        if overrides:
            config.update(overrides)

        units = config.pop("units")
        proj_size = config.pop("proj_size", None)

        self.profile = profile_obj
        self.config = {"units": units, "proj_size": proj_size, **config}
        self.model = CfC(
            input_size=input_size,
            units=units,
            proj_size=proj_size,
            **config,
        )

    def __call__(
        self,
        inputs: mx.array,
        hx: Optional[mx.array] = None,
        timespans: Optional[mx.array] = None,
        *,
        return_state: bool = False,
    ):
        outputs, state = self.model(inputs, hx=hx, timespans=timespans)
        if return_state:
            return outputs, state
        return outputs

    def apply_constraints(self) -> None:
        if hasattr(self.model, "rnn_cell") and hasattr(self.model.rnn_cell, "apply_weight_constraints"):
            self.model.rnn_cell.apply_weight_constraints()
