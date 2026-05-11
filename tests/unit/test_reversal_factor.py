from __future__ import annotations

import pandas as pd
import pytest

from quant_research.factors import FactorContext, FactorEngine
from quant_research.factors.library import FiveMinuteReversalFactor
from quant_research.signals import SignalGenerator, SignalSpec
from quant_research.strategies import (
    FiveMinuteCrossSectionalConfig,
    FiveMinuteCrossSectionalStrategy,
)


def test_five_minute_reversal_factor_scores_recent_losers_higher() -> None:
    context = FactorContext(
        data=_FakeFiveMinuteDataPortal(),
        start="2024-01-02T09:35:00+08:00",
        end="2024-01-02T09:45:00+08:00",
        symbols=("600000.SH", "000001.SZ"),
        market="CN",
        asset_type="equity",
        frequency="5m",
    )

    result = FactorEngine().compute(
        FiveMinuteReversalFactor(
            name="baseline_a_5m_reversal",
            inputs=("close_price",),
            lookback_bars=1,
        ),
        context,
    )

    latest = result.frame.loc[
        result.frame["timestamp"] == "2024-01-02T09:45:00+08:00"
    ].sort_values("instrument_id")
    assert latest["factor_value"].tolist() == pytest.approx([0.05, -0.05])


def test_five_minute_reversal_factor_applies_turnover_filter() -> None:
    context = FactorContext(
        data=_FakeFiveMinuteDataPortal(),
        start="2024-01-02T09:35:00+08:00",
        end="2024-01-02T09:45:00+08:00",
        symbols=("600000.SH", "000001.SZ"),
        market="CN",
        asset_type="equity",
        frequency="5m",
    )

    result = FactorEngine().compute(
        FiveMinuteReversalFactor(
            name="baseline_a_5m_reversal",
            inputs=("close_price", "turnover"),
            lookback_bars=1,
            liquidity_window_bars=2,
            min_avg_turnover=1_000_000.0,
        ),
        context,
    )

    assert result.frame["instrument_id"].unique().tolist() == ["inst-strong-up"]
    assert "avg_turnover" in result.frame.columns


def test_five_minute_reversal_factor_requires_5m_context() -> None:
    context = FactorContext(
        data=_FakeFiveMinuteDataPortal(),
        start="2024-01-02T09:35:00+08:00",
        end="2024-01-02T09:45:00+08:00",
        symbols=("600000.SH",),
        market="CN",
        asset_type="equity",
        frequency="1m",
    )

    with pytest.raises(ValueError, match="5m"):
        FiveMinuteReversalFactor(
            name="baseline_a_5m_reversal",
            inputs=("close_price",),
        ).compute(context)


def test_baseline_a_reversal_pipeline_selects_recent_loser() -> None:
    context = FactorContext(
        data=_FakeFiveMinuteDataPortal(),
        start="2024-01-02T09:35:00+08:00",
        end="2024-01-02T09:45:00+08:00",
        symbols=("600000.SH", "000001.SZ"),
        market="CN",
        asset_type="equity",
        frequency="5m",
    )
    factor_result = FactorEngine().compute(
        FiveMinuteReversalFactor(
            name="baseline_a_5m_reversal",
            inputs=("close_price",),
            lookback_bars=1,
        ),
        context,
    )
    signals = SignalGenerator().generate(
        factor_result.frame,
        SignalSpec(
            name="baseline_a_5m_reversal_rank",
            factor_name="baseline_a_5m_reversal",
            method="rank",
            parameters={"ascending": True, "percentile": True},
        ),
    )

    strategy_result = FiveMinuteCrossSectionalStrategy().build(
        signals.frame,
        FiveMinuteCrossSectionalConfig(
            name="baseline_a_5m_main_board",
            top_n=1,
            weighting="equal",
        ),
        current_positions=pd.DataFrame(
            columns=["instrument_id", "current_weight", "sellable_weight"]
        ),
    )

    latest_selection = strategy_result.selected_signals.loc[
        strategy_result.selected_signals["timestamp"] == "2024-01-02T09:45:00+08:00"
    ]
    assert latest_selection["instrument_id"].tolist() == ["inst-recent-loser"]


class _FakeFiveMinuteDataPortal:
    def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
        _ = args, kwargs
        return pd.DataFrame(
            [
                {
                    "instrument_id": "inst-recent-loser",
                    "bar_end_time": "2024-01-02T09:35:00+08:00",
                    "close_price": 100.0,
                    "turnover": 500_000.0,
                },
                {
                    "instrument_id": "inst-recent-loser",
                    "bar_end_time": "2024-01-02T09:40:00+08:00",
                    "close_price": 100.0,
                    "turnover": 500_000.0,
                },
                {
                    "instrument_id": "inst-recent-loser",
                    "bar_end_time": "2024-01-02T09:45:00+08:00",
                    "close_price": 95.0,
                    "turnover": 500_000.0,
                },
                {
                    "instrument_id": "inst-strong-up",
                    "bar_end_time": "2024-01-02T09:35:00+08:00",
                    "close_price": 100.0,
                    "turnover": 2_000_000.0,
                },
                {
                    "instrument_id": "inst-strong-up",
                    "bar_end_time": "2024-01-02T09:40:00+08:00",
                    "close_price": 100.0,
                    "turnover": 2_000_000.0,
                },
                {
                    "instrument_id": "inst-strong-up",
                    "bar_end_time": "2024-01-02T09:45:00+08:00",
                    "close_price": 105.0,
                    "turnover": 2_000_000.0,
                },
            ]
        )
