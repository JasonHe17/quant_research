"""Signal model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Configuration for transforming factors into signals."""

    name: str
    factor_name: str
    method: str = "identity"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name is required")
        if not self.factor_name:
            raise ValueError("factor_name is required")
        if self.method not in {"identity", "rank", "threshold"}:
            raise ValueError("method must be identity, rank, or threshold")


@dataclass(frozen=True, slots=True)
class SignalResult:
    """Signal generation output."""

    spec: SignalSpec
    frame: pd.DataFrame
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts: dict[str, str] = field(default_factory=dict)

    def with_artifacts(self, artifacts: dict[str, str]) -> "SignalResult":
        return SignalResult(
            spec=self.spec,
            frame=self.frame,
            diagnostics=self.diagnostics,
            artifacts={**self.artifacts, **artifacts},
        )
