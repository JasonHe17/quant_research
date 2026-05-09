"""Universe model definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    """Date-aware research universe specification."""

    name: str
    symbols: tuple[str, ...]
    market: str | None = None
    asset_type: str | None = None
