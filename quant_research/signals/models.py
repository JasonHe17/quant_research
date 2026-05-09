"""Signal model definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Configuration for transforming factors into signals."""

    name: str
    factor_name: str
    method: str
