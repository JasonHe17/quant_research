"""Portfolio construction models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Configuration for target portfolio construction."""

    name: str
    rebalance_frequency: str = "1d"
    weighting: str = "equal"
    max_weight: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("portfolio name is required")
        if self.weighting not in {"equal", "signal"}:
            raise ValueError("weighting must be 'equal' or 'signal'")
        if self.max_weight is not None and not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class PortfolioConstructionResult:
    """Portfolio construction output tables."""

    config: PortfolioConfig
    target_weights: pd.DataFrame
    rebalance_orders: pd.DataFrame
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts: dict[str, str] = field(default_factory=dict)

    def with_artifacts(
        self, artifacts: dict[str, str]
    ) -> "PortfolioConstructionResult":
        return PortfolioConstructionResult(
            config=self.config,
            target_weights=self.target_weights,
            rebalance_orders=self.rebalance_orders,
            diagnostics=self.diagnostics,
            artifacts={**self.artifacts, **artifacts},
        )
