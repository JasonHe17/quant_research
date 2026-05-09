"""Base factor interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class FactorContext:
    """Runtime context passed into factor calculations."""

    data: Any
    start: str
    end: str
    symbols: tuple[str, ...]
    market: str | None = None
    asset_type: str | None = None
    frequency: str = "1m"
    snapshot: str | None = None
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FactorResult:
    """Computed factor output plus reproducibility metadata."""

    factor_name: str
    frame: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Factor:
    """Declarative factor definition."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, context: FactorContext) -> pd.DataFrame:
        raise NotImplementedError("factor computation is implemented by subclasses")
