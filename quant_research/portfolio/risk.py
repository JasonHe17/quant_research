"""Portfolio risk model placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskConstraint:
    """One portfolio risk constraint."""

    name: str
    limit: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name is required")
        if self.limit < 0:
            raise ValueError("constraint limit must be non-negative")
