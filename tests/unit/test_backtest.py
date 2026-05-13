from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestFrames,
    ExecutionModel,
    TargetWeightExecutionConfig,
)


def test_backtest_engine_runs_simulator_and_computes_metrics() -> None:
    config = BacktestConfig(
        name="smoke",
        start="2024-01-01",
        end="2024-01-03",
        data_snapshot="2026-05-09",
        initial_cash=100.0,
    )

    result = BacktestEngine().run(config, _simulator)

    assert result.config == config
    assert result.metrics["total_return"] == pytest.approx(0.2)
    assert result.metrics["max_drawdown"] == pytest.approx(-1.0 / 12.0)
    assert list(result.trades.columns) == [
        "timestamp",
        "instrument_id",
        "quantity",
        "price",
    ]


def test_backtest_engine_runs_target_weight_execution_core() -> None:
    config = BacktestConfig(
        name="target-weight",
        start="2025-01-03T09:30:00+08:00",
        end="2025-01-03T09:35:00+08:00",
        data_snapshot="2026-05-09",
        initial_cash=1_000.0,
        frequency="5m",
    )
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-1",
                "open_price": 10.0,
                "close_price": 11.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 1.0,
            }
        ]
    )

    result = BacktestEngine().run_target_weight(
        config,
        executions,
        TargetWeightExecutionConfig(initial_cash=1_000.0, lot_size=1),
    )

    assert result.trades.loc[0, "quantity"] == 100
    assert result.positions.loc[0, "market_value"] == 1_100.0
    assert result.metrics["total_return"] == pytest.approx(0.1)
    assert result.diagnostics.loc[0, "trade_count"] == 1


def test_backtest_engine_persists_artifacts(tmp_path: Path) -> None:
    config = BacktestConfig(
        name="smoke",
        start="2024-01-01",
        end="2024-01-03",
        data_snapshot="2026-05-09",
        initial_cash=100.0,
    )
    store = ArtifactStore.from_path(tmp_path)

    result = BacktestEngine(artifact_store=store).run(
        config, _simulator, persist=True
    )

    assert set(result.artifacts) == {
        "trades",
        "positions",
        "equity_curve",
        "diagnostics",
    }
    assert store.read_backtest_artifact("smoke", "equity_curve").equals(
        result.equity_curve
    )


def test_backtest_engine_rejects_missing_required_columns() -> None:
    config = BacktestConfig(
        name="bad",
        start="2024-01-01",
        end="2024-01-03",
        data_snapshot="2026-05-09",
    )

    with pytest.raises(ValueError, match="trades"):
        BacktestEngine().run(config, _bad_simulator)


def test_backtest_engine_requires_artifact_store_when_persisting() -> None:
    config = BacktestConfig(
        name="smoke",
        start="2024-01-01",
        end="2024-01-03",
        data_snapshot="2026-05-09",
    )

    with pytest.raises(ValueError, match="artifact_store"):
        BacktestEngine().run(config, _simulator, persist=True)


def test_backtest_config_and_execution_model_validate_inputs() -> None:
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestConfig(
            name="bad",
            start="2024-01-01",
            end="2024-01-03",
            data_snapshot="2026-05-09",
            initial_cash=0,
        )

    with pytest.raises(ValueError, match="slippage"):
        ExecutionModel(name="bad", slippage_bps=-1.0)


def _simulator(config: BacktestConfig) -> BacktestFrames:
    _ = config
    return BacktestFrames(
        trades=pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-01",
                    "instrument_id": "inst-600000",
                    "quantity": 10.0,
                    "price": 10.0,
                }
            ]
        ),
        positions=pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-01",
                    "instrument_id": "inst-600000",
                    "quantity": 10.0,
                    "market_value": 100.0,
                }
            ]
        ),
        equity_curve=pd.DataFrame(
            [
                {"timestamp": "2024-01-01", "equity": 100.0},
                {"timestamp": "2024-01-02", "equity": 120.0},
                {"timestamp": "2024-01-03", "equity": 110.0},
                {"timestamp": "2024-01-04", "equity": 120.0},
            ]
        ),
    )


def _bad_simulator(config: BacktestConfig) -> BacktestFrames:
    _ = config
    return BacktestFrames(
        trades=pd.DataFrame([{"timestamp": "2024-01-01"}]),
        positions=pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-01",
                    "instrument_id": "inst-600000",
                    "quantity": 10.0,
                    "market_value": 100.0,
                }
            ]
        ),
        equity_curve=pd.DataFrame([{"timestamp": "2024-01-01", "equity": 100.0}]),
    )
