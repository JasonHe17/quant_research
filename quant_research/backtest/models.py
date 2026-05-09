"""Backtest configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for one reproducible backtest run."""

    name: str
    start: str
    end: str
    data_snapshot: str
    initial_cash: float = 1_000_000.0
    frequency: str = "1d"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backtest name is required")
        if not self.start:
            raise ValueError("start is required")
        if not self.end:
            raise ValueError("end is required")
        if not self.data_snapshot:
            raise ValueError("data_snapshot is required")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")


@dataclass(frozen=True, slots=True)
class BacktestFrames:
    """Tabular outputs produced by a backtest simulator."""

    trades: pd.DataFrame
    positions: pd.DataFrame
    equity_curve: pd.DataFrame
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest outputs plus derived metrics and artifact references."""

    config: BacktestConfig
    trades: pd.DataFrame
    positions: pd.DataFrame
    equity_curve: pd.DataFrame
    diagnostics: pd.DataFrame
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)

    def with_artifacts(self, artifacts: dict[str, str]) -> "BacktestResult":
        return BacktestResult(
            config=self.config,
            trades=self.trades,
            positions=self.positions,
            equity_curve=self.equity_curve,
            diagnostics=self.diagnostics,
            metrics=dict(self.metrics),
            artifacts={**self.artifacts, **artifacts},
        )
