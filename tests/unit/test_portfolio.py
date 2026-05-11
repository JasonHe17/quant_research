from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.portfolio import (
    PortfolioConfig,
    PortfolioConstructor,
    RiskConstraint,
    apply_cn_t1_constraints,
)


def test_portfolio_constructor_builds_equal_weight_targets() -> None:
    constructor = PortfolioConstructor()

    result = constructor.build(_signals(), PortfolioConfig(name="equal-test"))

    assert result.target_weights["target_weight"].tolist() == [0.5, 0.5]
    assert result.rebalance_orders["delta_weight"].tolist() == [0.5, 0.5]
    assert result.diagnostics.loc[0, "instrument_count"] == 2
    assert result.diagnostics.loc[0, "gross_weight"] == 1.0


def test_portfolio_constructor_builds_signal_weight_targets_with_current_positions() -> None:
    constructor = PortfolioConstructor()
    current = pd.DataFrame(
        [
            {"instrument_id": "inst-1", "current_weight": 0.2},
            {"instrument_id": "inst-2", "current_weight": 0.1},
        ]
    )

    result = constructor.build(
        _signals(),
        PortfolioConfig(name="signal-test", weighting="signal"),
        current_positions=current,
    )

    assert result.target_weights["target_weight"].tolist() == pytest.approx(
        [1.0 / 3.0, 2.0 / 3.0]
    )
    assert result.rebalance_orders["delta_weight"].tolist() == pytest.approx(
        [1.0 / 3.0 - 0.2, 2.0 / 3.0 - 0.1]
    )


def test_portfolio_constructor_respects_max_weight_clip() -> None:
    result = PortfolioConstructor().build(
        _signals(),
        PortfolioConfig(name="clipped", weighting="signal", max_weight=0.4),
    )

    assert result.target_weights["target_weight"].tolist() == [
        pytest.approx(1 / 3),
        0.4,
    ]


def test_portfolio_constructor_liquidates_positions_missing_from_targets() -> None:
    current = pd.DataFrame(
        [
            {"instrument_id": "inst-1", "current_weight": 0.4},
            {"instrument_id": "inst-old", "current_weight": 0.3},
        ]
    )

    result = PortfolioConstructor().build(
        pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-01T09:35:00+08:00",
                    "instrument_id": "inst-1",
                    "signal": 1.0,
                }
            ]
        ),
        PortfolioConfig(name="switch", weighting="equal"),
        current_positions=current,
    )

    old_order = result.rebalance_orders.loc[
        result.rebalance_orders["instrument_id"] == "inst-old"
    ].iloc[0]
    assert old_order["target_weight"] == 0.0
    assert old_order["delta_weight"] == pytest.approx(-0.3)


def test_cn_t1_constraints_cap_sells_to_sellable_weight() -> None:
    orders = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-1",
                "current_weight": 0.5,
                "target_weight": 0.0,
                "delta_weight": -0.5,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-2",
                "current_weight": 0.0,
                "target_weight": 0.2,
                "delta_weight": 0.2,
            },
        ]
    )
    current = pd.DataFrame(
        [
            {"instrument_id": "inst-1", "sellable_weight": 0.2},
            {"instrument_id": "inst-2", "sellable_weight": 0.0},
        ]
    )

    constrained = apply_cn_t1_constraints(orders, current_positions=current)

    sell = constrained.loc[constrained["instrument_id"] == "inst-1"].iloc[0]
    buy = constrained.loc[constrained["instrument_id"] == "inst-2"].iloc[0]
    assert sell["executable_delta_weight"] == pytest.approx(-0.2)
    assert sell["blocked_sell_weight"] == pytest.approx(0.3)
    assert sell["t1_blocked"]
    assert buy["executable_delta_weight"] == pytest.approx(0.2)
    assert buy["blocked_sell_weight"] == 0.0


def test_portfolio_constructor_persists_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore.from_path(tmp_path)
    result = PortfolioConstructor(artifact_store=store).build(
        _signals(),
        PortfolioConfig(name="persisted"),
        persist=True,
    )

    assert set(result.artifacts) == {
        "target_weights",
        "rebalance_orders",
        "diagnostics",
    }
    assert store.read_portfolio_artifact("persisted", "target_weights").equals(
        result.target_weights
    )


def test_portfolio_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        PortfolioConstructor().build(
            pd.DataFrame([{"instrument_id": "inst-1"}]),
            PortfolioConfig(name="bad"),
        )

    with pytest.raises(ValueError, match="artifact_store"):
        PortfolioConstructor().build(
            _signals(),
            PortfolioConfig(name="bad"),
            persist=True,
        )


def test_portfolio_config_and_risk_constraint_validate_inputs() -> None:
    with pytest.raises(ValueError, match="weighting"):
        PortfolioConfig(name="bad", weighting="optimizer")

    with pytest.raises(ValueError, match="max_weight"):
        PortfolioConfig(name="bad", max_weight=1.5)

    with pytest.raises(ValueError, match="limit"):
        RiskConstraint(name="turnover", limit=-1.0)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": "2024-01-01", "instrument_id": "inst-1", "signal": 1.0},
            {"timestamp": "2024-01-01", "instrument_id": "inst-2", "signal": 2.0},
        ]
    )
