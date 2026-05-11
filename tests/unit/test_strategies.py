from __future__ import annotations

import pandas as pd
import pytest

from quant_research.strategies import (
    FiveMinuteCrossSectionalConfig,
    FiveMinuteCrossSectionalStrategy,
)


def test_five_minute_cross_sectional_strategy_selects_top_signals_and_applies_t1() -> None:
    signals = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-new",
                "signal": 0.9,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-held",
                "signal": 0.2,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-low",
                "signal": 0.1,
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 0.5,
                "sellable_weight": 0.1,
            }
        ]
    )

    result = FiveMinuteCrossSectionalStrategy().build(
        signals,
        FiveMinuteCrossSectionalConfig(
            name="cn-5m",
            top_n=1,
            weighting="equal",
            max_weight=1.0,
        ),
        current_positions=current,
    )

    assert result.selected_signals["instrument_id"].tolist() == ["inst-new"]
    assert result.target_weights["target_weight"].tolist() == [1.0]
    held_order = result.constrained_orders.loc[
        result.constrained_orders["instrument_id"] == "inst-held"
    ].iloc[0]
    assert held_order["desired_delta_weight"] == pytest.approx(-0.5)
    assert held_order["executable_delta_weight"] == pytest.approx(-0.1)
    assert held_order["blocked_sell_weight"] == pytest.approx(0.4)
    assert result.diagnostics.loc[0, "blocked_sell_weight"] == pytest.approx(0.4)


def test_five_minute_cross_sectional_strategy_requires_explicit_parameters() -> None:
    with pytest.raises(ValueError, match="top_n"):
        FiveMinuteCrossSectionalConfig(name="bad", top_n=0, weighting="equal")

    with pytest.raises(ValueError, match="weighting"):
        FiveMinuteCrossSectionalConfig(name="bad", top_n=1, weighting="bad")  # type: ignore[arg-type]


def test_five_minute_cross_sectional_strategy_rejects_negative_signal_weights() -> None:
    signals = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-1",
                "signal": -0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="non-negative"):
        FiveMinuteCrossSectionalStrategy().build(
            signals,
            FiveMinuteCrossSectionalConfig(
                name="cn-5m",
                top_n=1,
                weighting="signal",
            ),
            current_positions=pd.DataFrame(
                columns=["instrument_id", "current_weight", "sellable_weight"]
            ),
        )
