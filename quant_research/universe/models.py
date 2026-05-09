"""Universe model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    """Date-aware research universe specification."""

    name: str
    symbols: tuple[str, ...]
    market: str | None = None
    asset_type: str | None = None
    start: str | None = None
    end: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("universe name is required")
        if not self.symbols:
            raise ValueError("symbols are required")


@dataclass(frozen=True, slots=True)
class Universe:
    """Resolved date-aware universe membership."""

    spec: UniverseSpec
    members: pd.DataFrame
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts: dict[str, str] = field(default_factory=dict)

    def with_artifacts(self, artifacts: dict[str, str]) -> "Universe":
        return Universe(
            spec=self.spec,
            members=self.members,
            diagnostics=self.diagnostics,
            artifacts={**self.artifacts, **artifacts},
        )
