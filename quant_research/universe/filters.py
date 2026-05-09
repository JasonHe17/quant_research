"""Universe filter placeholders."""

from __future__ import annotations

from quant_research.universe.models import UniverseSpec


def identity_filter(universe: UniverseSpec) -> UniverseSpec:
    """Return a universe unchanged."""

    return universe
