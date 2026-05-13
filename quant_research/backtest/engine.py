"""Backtest orchestration."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.backtest.cn_equity import (
    TargetWeightExecutionConfig,
    TargetWeightExecutionSimulator,
)
from quant_research.backtest.models import (
    BacktestConfig,
    BacktestFrames,
    BacktestResult,
)
from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown
from quant_research.schemas import validate_standard_table


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

    def run_target_weight(
        self,
        config: BacktestConfig,
        executions: pd.DataFrame,
        execution_config: TargetWeightExecutionConfig,
        *,
        persist: bool = False,
    ) -> BacktestResult:
        """Run the built-in target-weight execution simulator."""

        if execution_config.initial_cash != config.initial_cash:
            raise ValueError("execution_config.initial_cash must match BacktestConfig")
        simulator = TargetWeightExecutionSimulator(execution_config)
        trades, equity_curve, final_positions, diagnostics = (
            simulator.run_with_diagnostics(executions)
        )
        frames = BacktestFrames(
            trades=_trades_for_backtest(trades),
            positions=_positions_for_backtest(
                final_positions,
                timestamp=_last_timestamp(equity_curve, default=config.end),
                last_prices=simulator.state.last_prices,
            ),
            equity_curve=_equity_curve_with_initial_row(equity_curve, config=config),
            diagnostics=diagnostics,
        )
        _validate_frames(frames)
        metrics = _metrics_from_equity_curve(frames.equity_curve)
        result = BacktestResult(
            config=config,
            trades=frames.trades,
            positions=frames.positions,
            equity_curve=frames.equity_curve,
            diagnostics=frames.diagnostics,
            metrics=metrics,
        )
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return result.with_artifacts(self.artifact_store.write_backtest(result))
        return result


def _validate_frames(frames: BacktestFrames) -> None:
    validate_standard_table("backtest_trades", frames.trades)
    validate_standard_table("backtest_positions", frames.positions)
    validate_standard_table("backtest_equity_curve", frames.equity_curve)
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


def _trades_for_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp", "instrument_id", "quantity", "price"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    output = trades.copy()
    if "quantity" not in output.columns and "shares" in output.columns:
        output["quantity"] = output["shares"]
    return output.loc[:, [*columns, *[c for c in output.columns if c not in columns]]]


def _positions_for_backtest(
    final_positions: pd.DataFrame,
    *,
    timestamp: object,
    last_prices: dict[str, float],
) -> pd.DataFrame:
    columns = ["timestamp", "instrument_id", "quantity", "market_value"]
    if final_positions.empty:
        return pd.DataFrame(columns=columns)
    output = final_positions.copy()
    output["timestamp"] = timestamp
    output["quantity"] = output["shares"].astype(float)
    output["market_value"] = output.apply(
        lambda row: float(row["quantity"]) * last_prices.get(str(row["instrument_id"]), 0.0),
        axis=1,
    )
    return output.loc[:, [*columns, *[c for c in output.columns if c not in columns]]]


def _equity_curve_with_initial_row(
    equity_curve: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    initial = pd.DataFrame(
        [
            {
                "timestamp": config.start,
                "cash": config.initial_cash,
                "positions_value": 0.0,
                "equity": config.initial_cash,
            }
        ]
    )
    if equity_curve.empty:
        return initial
    return pd.concat([initial, equity_curve.copy()], ignore_index=True)


def _last_timestamp(equity_curve: pd.DataFrame, *, default: object) -> object:
    if equity_curve.empty:
        return default
    return equity_curve.iloc[-1]["timestamp"]
