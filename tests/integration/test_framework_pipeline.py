from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.backtest import BacktestConfig, BacktestEngine, BacktestFrames
from quant_research.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentRunStore,
)
from quant_research.factors import Factor, FactorContext, FactorEngine
from quant_research.metrics import MetricsEngine
from quant_research.portfolio import PortfolioConfig, PortfolioConstructor
from quant_research.signals import SignalGenerator, SignalSpec
from quant_research.universe import UniverseBuilder, UniverseSpec, active_on


class CloseReturnFactor(Factor):
    def compute(self, context: FactorContext) -> pd.DataFrame:
        bars = context.data.get_bars(
            list(context.symbols),
            start=context.start,
            end=context.end,
            frequency=context.frequency,
            adjustment="raw",
            market=context.market,
            fields=["instrument_id", "bar_end_time", "close_price"],
            cache=False,
        )
        frame = bars.rename(columns={"bar_end_time": "timestamp"}).copy()
        frame["factor_value"] = frame["close_price"].pct_change().fillna(0.0)
        return frame.loc[:, ["instrument_id", "timestamp", "factor_value"]]


def test_framework_pipeline_smoke(tmp_path: Path) -> None:
    artifact_store = ArtifactStore.from_path(tmp_path / "research_store")
    data = _FakeDataPortal()

    universe = UniverseBuilder(artifact_store=artifact_store).build(
        UniverseSpec(
            name="cn-core",
            symbols=("600000.SH",),
            market="CN",
            asset_type="equity",
            start="2024-01-01",
            end="2024-12-31",
        ),
        data=data,
        persist=True,
    )
    active_universe = active_on(universe, "2024-01-02")

    factor_result = FactorEngine(artifact_store=artifact_store).compute(
        CloseReturnFactor(name="close_return", inputs=("close_price",)),
        FactorContext(
            data=data,
            start="2024-01-02T09:31:00+08:00",
            end="2024-01-02T09:32:00+08:00",
            symbols=tuple(active_universe.members["symbol"].tolist()),
            market="CN",
            asset_type="equity",
            frequency="1m",
            snapshot="2026-05-09",
        ),
        persist=True,
    )

    signal_result = SignalGenerator(artifact_store=artifact_store).generate(
        factor_result.frame,
        SignalSpec(
            name="ranked_close_return",
            factor_name="close_return",
            method="rank",
            parameters={"ascending": False},
        ),
        persist=True,
    )

    portfolio_result = PortfolioConstructor(artifact_store=artifact_store).build(
        signal_result.frame,
        PortfolioConfig(name="ranked_portfolio", weighting="signal", max_weight=1.0),
        persist=True,
    )

    backtest_result = BacktestEngine(artifact_store=artifact_store).run(
        BacktestConfig(
            name="pipeline_backtest",
            start="2024-01-02",
            end="2024-01-03",
            data_snapshot="2026-05-09",
            initial_cash=100.0,
        ),
        _simulator_from_portfolio(portfolio_result.target_weights),
        persist=True,
    )

    report = MetricsEngine(artifact_store=artifact_store).from_equity_curve(
        backtest_result.equity_curve,
        name="pipeline_report",
        metadata={"data_snapshot": "2026-05-09"},
        persist=True,
    )

    run_store = ExperimentRunStore(root=tmp_path / "research_store")
    runner = ExperimentRunner(run_store=run_store)
    run = runner.create_run(
        ExperimentConfig(
            name="pipeline_smoke",
            data_snapshot="2026-05-09",
            parameters={"symbols": ["600000.SH"]},
        )
    )
    completed = runner.complete_run(
        run,
        artifacts={
            "universe": universe.artifacts["members"],
            "factor": str(artifact_store.factor_path("close_return")),
            "signals": signal_result.artifacts["signals"],
            "portfolio": portfolio_result.artifacts["target_weights"],
            "backtest": backtest_result.artifacts["equity_curve"],
            "report": report.artifacts["metrics_report"],
        },
        metrics={metric.name: metric.value for metric in report.metrics},
    )

    assert len(active_universe.members) == 1
    assert factor_result.frame["factor_name"].unique().tolist() == ["close_return"]
    assert signal_result.frame["signal_name"].unique().tolist() == [
        "ranked_close_return"
    ]
    assert not portfolio_result.rebalance_orders.empty
    assert backtest_result.metrics["total_return"] == pytest.approx(0.2)
    assert report.artifacts["metrics_report"].endswith("metrics.json")
    assert run_store.read(completed.run_id) == completed


def _simulator_from_portfolio(target_weights: pd.DataFrame):
    def simulator(config: BacktestConfig) -> BacktestFrames:
        instrument_id = str(target_weights.iloc[0]["instrument_id"])
        return BacktestFrames(
            trades=pd.DataFrame(
                [
                    {
                        "timestamp": config.start,
                        "instrument_id": instrument_id,
                        "quantity": 10.0,
                        "price": 10.0,
                    }
                ]
            ),
            positions=pd.DataFrame(
                [
                    {
                        "timestamp": config.start,
                        "instrument_id": instrument_id,
                        "quantity": 10.0,
                        "market_value": 100.0,
                    }
                ]
            ),
            equity_curve=pd.DataFrame(
                [
                    {"timestamp": config.start, "equity": 100.0},
                    {"timestamp": config.end, "equity": 120.0},
                ]
            ),
        )

    return simulator


class _FakeDataPortal:
    def resolve_instruments(
        self,
        symbols: list[str],
        *,
        market: str | None = None,
        asset_type: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical_code": symbol,
                    "instrument_id": "inst-600000",
                    "market": market,
                    "asset_type": asset_type,
                }
                for symbol in symbols
            ]
        )

    def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "instrument_id": "inst-600000",
                    "bar_end_time": "2024-01-02T09:31:00+08:00",
                    "close_price": 10.0,
                },
                {
                    "instrument_id": "inst-600000",
                    "bar_end_time": "2024-01-02T09:32:00+08:00",
                    "close_price": 11.0,
                },
            ]
        )
