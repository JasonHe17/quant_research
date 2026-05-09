"""Portfolio risk model placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskConstraint:
    """One portfolio risk constraint."""

    name: str
    limit: float
