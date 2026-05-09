"""Backtest orchestration."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.backtest.models import (
    BacktestConfig,
    BacktestFrames,
    BacktestResult,
)
from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown


BacktestSimulator = Callable[[BacktestConfig], BacktestFrames]


class BacktestEngine:
    """Orchestrates simulator execution, validation, metrics, and persistence."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def run(
        self,
        config: BacktestConfig,
        simulator: BacktestSimulator,
        *,
        persist: bool = False,
    ) -> BacktestResult:
        frames = simulator(config)
        _validate_frames(frames)
        metrics = _metrics_from_equity_curve(frames.equity_curve)
        result = BacktestResult(
            config=config,
            trades=frames.trades.copy(),
            positions=frames.positions.copy(),
            equity_curve=frames.equity_curve.copy(),
            diagnostics=frames.diagnostics.copy(),
            metrics=metrics,
        )
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return result.with_artifacts(self.artifact_store.write_backtest(result))
        return result


def _validate_frames(frames: BacktestFrames) -> None:
    _require_columns(
        frames.trades,
        ("timestamp", "instrument_id", "quantity", "price"),
        name="trades",
    )
    _require_columns(
        frames.positions,
        ("timestamp", "instrument_id", "quantity", "market_value"),
        name="positions",
    )
    _require_columns(frames.equity_curve, ("timestamp", "equity"), name="equity_curve")
    if frames.equity_curve.empty:
        raise ValueError("equity_curve must not be empty")


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, name: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _metrics_from_equity_curve(equity_curve: pd.DataFrame) -> dict[str, float]:
    values = [float(value) for value in equity_curve["equity"].tolist()]
    return {
        "total_return": total_return(values[0], values[-1]),
        "max_drawdown": max_drawdown(values),
    }
