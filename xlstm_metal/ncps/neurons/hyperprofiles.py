"""Hyperparameter profile loader for MLX cells."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HyperProfile:
    name: str
    description: str
    ode_unfolds: Optional[int]
    solver: Optional[str]
    input_mapping: Optional[str]
    initializers: Dict[str, float]
    constraints: Dict[str, float]
    extras: Dict[str, Any]


_PROFILE_CACHE: Dict[str, HyperProfile] = {}


def _profiles_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "profiles"


def load_profile(name: str) -> HyperProfile:
    """Load a hyperparameter profile by name (case insensitive)."""

    key = name.lower()
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]

    profile_path: Optional[Path] = None
    for candidate in _profiles_dir().glob("*.json"):
        if candidate.stem.lower() == key:
            profile_path = candidate
            break

    if profile_path is None:
        raise FileNotFoundError(f"Unknown hyperparameter profile '{name}'.")

    with profile_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    profile = HyperProfile(
        name=data["name"],
        description=data.get("description", ""),
        ode_unfolds=data.get("ode_unfolds"),
        solver=data.get("solver"),
        input_mapping=data.get("input_mapping"),
        initializers=data.get("initializers", {}),
        constraints=data.get("constraints", {}),
        extras=data.get("extras", {}),
    )
    _PROFILE_CACHE[key] = profile
    return profile


__all__ = ["HyperProfile", "load_profile"]
