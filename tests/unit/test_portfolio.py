from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.artifacts import ArtifactStore
from quant_research.portfolio import (
    PortfolioConfig,
    PortfolioConstructor,
    RiskConstraint,
    RollingRegimeGateConfig,
    apply_cn_t1_constraints,
    build_rolling_regime_gate,
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


def test_rolling_regime_gate_uses_only_lagged_observations() -> None:
    diagnostics = pd.DataFrame(
        [
            _regime_row("t0", top=-0.02, spread=-0.03, ic=-0.4),
            _regime_row("t1", top=0.03, spread=0.04, ic=0.5),
            _regime_row("t2", top=0.04, spread=0.05, ic=0.6),
        ]
    )

    schedule = build_rolling_regime_gate(
        diagnostics,
        RollingRegimeGateConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            reduced_scale=0.5,
            blocked_scale=0.0,
            block_top_return=-0.01,
            block_spread=-0.01,
            block_rank_ic=-0.1,
        ),
    )

    assert schedule["gate_reason"].tolist() == [
        "warmup",
        "blocked_exposure",
        "full_exposure",
    ]
    assert schedule["gross_exposure_scale"].tolist() == [1.0, 0.0, 1.0]
    assert schedule.loc[1, "rolling_score_top_n_mean_label"] == pytest.approx(-0.02)


def test_rolling_regime_gate_reduces_before_blocking_thresholds() -> None:
    diagnostics = pd.DataFrame(
        [
            _regime_row("t0", top=0.01, spread=0.01, ic=0.1),
            _regime_row("t1", top=0.002, spread=-0.0005, ic=0.02),
            _regime_row("t2", top=0.01, spread=0.01, ic=0.1),
        ]
    )

    schedule = build_rolling_regime_gate(
        diagnostics,
        RollingRegimeGateConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            reduced_scale=0.25,
            blocked_scale=0.0,
            min_spread=0.0,
            block_spread=-0.01,
            block_rank_ic=-0.1,
        ),
    )

    assert schedule.loc[1, "gate_reason"] == "full_exposure"
    assert schedule.loc[2, "gate_reason"] == "reduced_exposure"
    assert schedule.loc[2, "gross_exposure_scale"] == pytest.approx(0.25)


def test_rolling_regime_gate_confirms_states_and_limits_scale_steps() -> None:
    diagnostics = pd.DataFrame(
        [
            _regime_row("t0", top=-0.02, spread=-0.03, ic=-0.4),
            _regime_row("t1", top=-0.02, spread=-0.03, ic=-0.4),
            _regime_row("t2", top=0.03, spread=0.04, ic=0.5),
            _regime_row("t3", top=0.03, spread=0.04, ic=0.5),
            _regime_row("t4", top=0.03, spread=0.04, ic=0.5),
        ]
    )

    schedule = build_rolling_regime_gate(
        diagnostics,
        RollingRegimeGateConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            state_confirmation_windows=2,
            max_scale_change_per_window=0.5,
            blocked_scale=0.0,
            block_top_return=-0.01,
            block_spread=-0.01,
            block_rank_ic=-0.1,
        ),
    )

    assert schedule["raw_gate_reason"].tolist() == [
        "warmup",
        "blocked_exposure",
        "blocked_exposure",
        "full_exposure",
        "full_exposure",
    ]
    assert schedule["gate_reason"].tolist() == [
        "warmup",
        "warmup",
        "blocked_exposure",
        "blocked_exposure",
        "full_exposure",
    ]
    assert schedule["gross_exposure_scale"].tolist() == [1.0, 1.0, 0.5, 0.0, 0.5]
    assert schedule["scale_step_limited"].tolist() == [False, False, True, False, True]


def test_rolling_regime_gate_budget_mode_uses_continuous_scale() -> None:
    diagnostics = pd.DataFrame(
        [
            _regime_row("t0", top=0.02, spread=0.02, ic=0.2),
            _regime_row("t1", top=-0.001, spread=-0.001, ic=-0.05),
            _regime_row("t2", top=0.0, spread=0.0, ic=0.0),
            _regime_row("t3", top=0.001, spread=0.001, ic=0.05),
        ]
    )

    schedule = build_rolling_regime_gate(
        diagnostics,
        RollingRegimeGateConfig(
            gate_mode="budget",
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            budget_min_scale=0.25,
            budget_max_scale=1.0,
            budget_top_return_floor=-0.001,
            budget_top_return_ceiling=0.001,
            budget_spread_floor=-0.001,
            budget_spread_ceiling=0.001,
            budget_rank_ic_floor=-0.05,
            budget_rank_ic_ceiling=0.05,
            max_scale_change_per_window=0.5,
        ),
    )

    assert schedule["raw_gate_reason"].tolist() == [
        "warmup",
        "budget_exposure",
        "budget_exposure",
        "budget_exposure",
    ]
    assert schedule["raw_gross_exposure_scale"].tolist() == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(0.25),
        pytest.approx(0.625),
    ]
    assert schedule["gross_exposure_scale"].tolist() == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(0.5),
        pytest.approx(0.625),
    ]
    assert schedule.loc[3, "budget_health_score"] == pytest.approx(0.5)


def test_rolling_regime_gate_budget_mode_supports_deadband_and_asymmetric_steps() -> None:
    diagnostics = pd.DataFrame(
        [
            _regime_row("t0", top=0.001, spread=0.001, ic=0.05),
            _regime_row("t1", top=0.0009, spread=0.0009, ic=0.045),
            _regime_row("t2", top=-0.001, spread=-0.001, ic=-0.05),
            _regime_row("t3", top=0.001, spread=0.001, ic=0.05),
            _regime_row("t4", top=0.001, spread=0.001, ic=0.05),
        ]
    )

    schedule = build_rolling_regime_gate(
        diagnostics,
        RollingRegimeGateConfig(
            gate_mode="budget",
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            budget_min_scale=0.25,
            budget_max_scale=1.0,
            budget_top_return_floor=-0.001,
            budget_top_return_ceiling=0.001,
            budget_spread_floor=-0.001,
            budget_spread_ceiling=0.001,
            budget_rank_ic_floor=-0.05,
            budget_rank_ic_ceiling=0.05,
            scale_change_deadband=0.05,
            max_scale_decrease_per_window=0.3,
            max_scale_increase_per_window=0.1,
        ),
    )

    assert schedule["raw_gross_exposure_scale"].tolist() == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(0.9625),
        pytest.approx(0.25),
        pytest.approx(1.0),
    ]
    assert schedule["gross_exposure_scale"].tolist() == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(0.7),
        pytest.approx(0.8),
    ]
    assert schedule["scale_deadband_held"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert schedule["scale_step_limited"].tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": "2024-01-01", "instrument_id": "inst-1", "signal": 1.0},
            {"timestamp": "2024-01-01", "instrument_id": "inst-2", "signal": 2.0},
        ]
    )


def _regime_row(timestamp: str, *, top: float, spread: float, ic: float) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "score_top_n_mean_label": top,
        "score_top_minus_bottom_label": spread,
        "score_rank_ic": ic,
    }
