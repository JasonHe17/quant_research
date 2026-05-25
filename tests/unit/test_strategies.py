from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pandas as pd
import pytest
import sys

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from quant_research.strategies import (
    CostAwareOptimizerConfig,
    CostAwareOptimizerPolicy,
    FiveMinuteCrossSectionalConfig,
    FiveMinuteCrossSectionalStrategy,
    RankBufferDropConfig,
    RankBufferDropPolicy,
)
from examples.run_tree_score_backtest import (
    TreeScoreBacktestParams,
    _bar_time_index,
    _build_segment_tree_score_executions,
    _build_tree_score_executions,
    _build_target_weights,
    _load_ranked_score_signals,
    _next_segment_end,
    _resolved_policy_estimated_cost_bps,
    _score_rank_limit,
    _run_tree_score_backtest_streaming,
)
import examples.run_baseline_a_real_backtest as baseline_backtest


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


def test_rank_buffer_drop_policy_holds_existing_name_until_exit_rank() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.8, 2),
        ]
    )
    state = pd.DataFrame(
        [{"instrument_id": "inst-held", "current_weight": 1.0, "holding_bars": 5}]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(target_count=1, entry_rank=1, exit_rank=2)
    ).decide(forecasts, state)

    assert result.policy_state["instrument_id"].tolist() == ["inst-held"]
    assert result.portfolio_intent.loc[
        result.portfolio_intent["instrument_id"] == "inst-held", "reason"
    ].iloc[0] == "hold_buffer"
    assert "inst-new" not in result.policy_state["instrument_id"].tolist()


def test_rank_buffer_drop_policy_exits_beyond_exit_rank_and_enters_replacement() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 3),
        ]
    )
    state = pd.DataFrame(
        [{"instrument_id": "inst-held", "current_weight": 1.0, "holding_bars": 5}]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(target_count=1, entry_rank=1, exit_rank=2)
    ).decide(forecasts, state)

    targets = result.trade_decisions.set_index("instrument_id")["target_weight"]
    reasons = result.trade_decisions.set_index("instrument_id")["decision_reason"]
    assert targets["inst-held"] == pytest.approx(0.0)
    assert targets["inst-new"] == pytest.approx(1.0)
    assert reasons["inst-held"] == "exit_rank"
    assert reasons["inst-new"] == "entry_rank"


def test_rank_buffer_drop_policy_uses_no_trade_band_for_small_resizes() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-a", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-b", 0.8, 2),
        ]
    )
    state = pd.DataFrame(
        [
            {"instrument_id": "inst-a", "current_weight": 0.49, "holding_bars": 5},
            {"instrument_id": "inst-b", "current_weight": 0.51, "holding_bars": 5},
        ]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(
            target_count=2,
            entry_rank=2,
            exit_rank=2,
            no_trade_weight_band=0.02,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-a", "target_weight"] == pytest.approx(0.49)
    assert decisions.loc["inst-b", "target_weight"] == pytest.approx(0.51)
    assert decisions["decision_reason"].tolist() == [
        "below_weight_band",
        "below_weight_band",
    ]
    assert result.order_intents.empty


def test_rank_buffer_drop_policy_blocks_unsellable_t1_reduction() -> None:
    forecasts = pd.DataFrame(
        [_forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 3)]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 1.0,
                "sellable_weight": 0.0,
                "holding_bars": 5,
            }
        ]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(target_count=1, entry_rank=1, exit_rank=2)
    ).decide(forecasts, state)

    decision = result.trade_decisions.iloc[0]
    assert decision["target_weight"] == pytest.approx(1.0)
    assert decision["decision_reason"] == "t1_sell_blocked"
    assert decision["constraint_flags"] == "sellable_weight_limited"


def test_rank_buffer_drop_policy_partially_rebalances_and_caps_turnover() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 3),
        ]
    )
    state = pd.DataFrame(
        [{"instrument_id": "inst-held", "current_weight": 1.0, "holding_bars": 5}]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(
            target_count=1,
            entry_rank=1,
            exit_rank=2,
            partial_rebalance_rate=0.5,
            max_gross_turnover_per_rebalance=0.5,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-held", "target_weight"] == pytest.approx(0.75)
    assert decisions.loc["inst-new", "target_weight"] == pytest.approx(0.25)
    assert set(result.trade_decisions["constraint_flags"]) == {"turnover_scaled"}


def test_rank_buffer_drop_policy_scales_new_entry_gross_exposure() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-a", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-b", 0.8, 2),
        ]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(
            target_count=2,
            entry_rank=2,
            exit_rank=2,
            gross_exposure_scale=0.5,
        )
    ).decide(forecasts)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-a", "target_weight"] == pytest.approx(0.25)
    assert decisions.loc["inst-b", "target_weight"] == pytest.approx(0.25)
    assert decisions["target_weight"].sum() == pytest.approx(0.5)
    assert set(decisions["constraint_flags"]) == {"gross_exposure_scaled"}
    assert result.diagnostics.loc[0, "target_gross_exposure"] == pytest.approx(0.5)
    assert result.diagnostics.loc[0, "gross_exposure_scaled_count"] == 2


def test_rank_buffer_drop_policy_marks_existing_reductions_as_risk_reduction() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-a", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-b", 0.8, 2),
        ]
    )
    state = pd.DataFrame(
        [
            {"instrument_id": "inst-a", "current_weight": 0.5, "holding_bars": 5},
            {"instrument_id": "inst-b", "current_weight": 0.5, "holding_bars": 5},
        ]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(
            target_count=2,
            entry_rank=2,
            exit_rank=2,
            gross_exposure_scale=0.5,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-a", "target_weight"] == pytest.approx(0.25)
    assert decisions.loc["inst-b", "target_weight"] == pytest.approx(0.25)
    assert decisions["decision_reason"].tolist() == ["risk_reduction", "risk_reduction"]
    assert result.diagnostics.loc[0, "risk_reduction_count"] == 2


def test_cost_aware_optimizer_uses_net_edge_after_cost_and_risk_penalty() -> None:
    forecasts = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-good",
                "score": 0.5,
                "rank": 1,
                "risk_penalty_bps": 5.0,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-risky",
                "score": 0.8,
                "rank": 2,
                "risk_penalty_bps": 100.0,
            },
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            score_to_edge_bps=100.0,
            estimated_cost_bps=5.0,
            risk_penalty_multiplier=1.0,
        )
    ).decide(forecasts)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-good", "target_weight"] == pytest.approx(1.0)
    assert decisions.loc["inst-good", "decision_reason"] == "entry_rank"
    assert decisions.loc["inst-risky", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-risky", "decision_reason"] == "below_edge"


def test_cost_aware_optimizer_applies_turnover_and_t1_constraints() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 2),
                "risk_penalty_bps": 100.0,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 1.0,
                "sellable_weight": 0.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            max_gross_turnover_per_rebalance=0.5,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-held", "target_weight"] == pytest.approx(1.0)
    assert decisions.loc["inst-held", "decision_reason"] == "t1_sell_blocked"
    assert "turnover_budget_limited" in decisions.loc["inst-new", "constraint_flags"]
    assert result.diagnostics.loc[0, "target_gross_exposure"] <= 1.0 + 1e-12


def test_cost_aware_optimizer_exits_unselected_holdings_without_lingering() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-old", 0.1, 2),
                "risk_penalty_bps": 100.0,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-old",
                "current_weight": 0.4,
                "sellable_weight": 0.4,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            partial_rebalance_rate=0.5,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-old", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-old", "decision_reason"] == "exit_rank"
    assert result.diagnostics.loc[0, "target_gross_exposure"] <= 1.0


def test_cost_aware_optimizer_filters_small_deltas_after_turnover_scaling() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 2),
                "risk_penalty_bps": 100.0,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 1.0,
                "sellable_weight": 1.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            max_gross_turnover_per_rebalance=0.001,
            no_trade_weight_band=0.01,
        )
    ).decide(forecasts, state)

    assert result.order_intents.empty
    assert result.diagnostics.loc[0, "order_intent_count"] == 0
    assert set(result.trade_decisions["decision_reason"]) == {"turnover_budget_limited"}
    assert result.diagnostics.loc[0, "turnover_budget_limited_count"] == 2


def test_cost_aware_optimizer_small_delta_filter_preserves_gross_cap() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-old", 0.1, 2),
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-old",
                "current_weight": 1.0,
                "sellable_weight": 1.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            max_gross_turnover_per_rebalance=0.001,
            no_trade_weight_band=0.01,
        )
    ).decide(forecasts, state)

    assert result.diagnostics.loc[0, "target_gross_exposure"] <= 1.0 + 1e-12


def test_cost_aware_optimizer_prefers_existing_holding_within_switch_cost() -> None:
    forecasts = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-new",
                "score": 0.52,
                "rank": 1,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-held",
                "score": 0.50,
                "rank": 2,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 1.0,
                "sellable_weight": 1.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            score_to_edge_bps=100.0,
            estimated_cost_bps=5.0,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-held", "target_weight"] == pytest.approx(1.0)
    assert decisions.loc["inst-new", "target_weight"] == pytest.approx(0.0)


def test_cost_aware_optimizer_replaces_holding_when_edge_clears_switch_cost() -> None:
    forecasts = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-new",
                "score": 0.70,
                "rank": 1,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-held",
                "score": 0.50,
                "rank": 2,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 1.0,
                "sellable_weight": 1.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=1,
            candidate_rank=2,
            score_to_edge_bps=100.0,
            estimated_cost_bps=5.0,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-new", "target_weight"] == pytest.approx(1.0)
    assert decisions.loc["inst-held", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-new", "decision_reason"] == "entry_rank"
    assert decisions.loc["inst-held", "decision_reason"] == "exit_rank"


def test_cost_aware_optimizer_turnover_budget_prioritizes_best_entries() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-a", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-b", 0.8, 2),
            _forecast("2024-01-02T09:35:00+08:00", "inst-c", 0.7, 3),
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-old", 0.1, 4),
                "risk_penalty_bps": 100.0,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-old",
                "current_weight": 1.0,
                "sellable_weight": 1.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=3,
            candidate_rank=4,
            weighting="equal",
            max_gross_turnover_per_rebalance=2 / 3,
            no_trade_weight_band=0.01,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-a", "target_weight"] == pytest.approx(1 / 3)
    assert decisions.loc["inst-old", "target_weight"] == pytest.approx(2 / 3)
    assert decisions.loc["inst-b", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-c", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-b", "decision_reason"] == "turnover_budget_limited"
    assert decisions.loc["inst-c", "decision_reason"] == "turnover_budget_limited"
    assert result.diagnostics.loc[0, "planned_gross_turnover"] == pytest.approx(2 / 3)


def test_cost_aware_optimizer_budget_uses_net_edge_not_rank_only() -> None:
    forecasts = pd.DataFrame(
        [
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-rank-top", 0.9, 1),
                "expected_edge_bps": 10.0,
            },
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-edge-top", 0.8, 2),
                "expected_edge_bps": 90.0,
            },
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-old", 0.1, 3),
                "risk_penalty_bps": 100.0,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-old",
                "current_weight": 1.0,
                "sellable_weight": 1.0,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=3,
            candidate_rank=3,
            weighting="equal",
            max_gross_turnover_per_rebalance=2 / 3,
            no_trade_weight_band=0.01,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-edge-top", "target_weight"] == pytest.approx(1 / 3)
    assert decisions.loc["inst-rank-top", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-rank-top", "decision_reason"] == "turnover_budget_limited"
    assert decisions.loc["inst-old", "target_weight"] == pytest.approx(2 / 3)


def test_cost_aware_optimizer_caps_exposure_increase_separately_from_sell_budget() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new-a", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-new-b", 0.8, 2),
            {
                **_forecast("2024-01-02T09:35:00+08:00", "inst-old", 0.1, 3),
                "risk_penalty_bps": 100.0,
            },
        ]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-old",
                "current_weight": 0.4,
                "sellable_weight": 0.4,
                "holding_bars": 5,
            }
        ]
    )

    result = CostAwareOptimizerPolicy(
        CostAwareOptimizerConfig(
            target_count=2,
            candidate_rank=3,
            weighting="equal",
            max_gross_turnover_per_rebalance=0.4,
            max_gross_exposure_increase_per_rebalance=0.1,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-old", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-new-a", "target_weight"] == pytest.approx(0.5)
    assert decisions.loc["inst-new-b", "target_weight"] == pytest.approx(0.0)
    assert decisions.loc["inst-new-b", "decision_reason"] == "turnover_budget_limited"
    assert result.diagnostics.loc[0, "target_gross_exposure"] == pytest.approx(0.5)
    assert result.diagnostics.loc[0, "planned_gross_turnover"] == pytest.approx(0.9)


def _forecast(timestamp: str, instrument_id: str, score: float, rank: int) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "score": score,
        "rank": rank,
    }


def test_tree_score_target_builder_supports_rank_buffer_drop_policy(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
            {"signal_time": "t1", "instrument_id": "inst-b", "score": 0.9, "rank": 1},
            {"signal_time": "t1", "instrument_id": "inst-a", "score": 0.8, "rank": 2},
        ]
    )

    result = _build_target_weights(ranked, _tree_score_params(tmp_path))

    by_time = {
        time: group["instrument_id"].tolist()
        for time, group in result.target_weights.groupby("signal_time", sort=True)
    }
    assert by_time == {"t0": ["inst-a"], "t1": ["inst-a"]}
    assert result.diagnostics["hold_count"].sum() == 1
    assert result.diagnostics["planned_gross_turnover"].sum() == pytest.approx(1.0)


def test_tree_score_target_builder_applies_timestamp_gross_exposure_schedule(
    tmp_path,
) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
            {"signal_time": "t1", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t1", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
        ]
    )
    schedule_path = tmp_path / "gross_exposure_schedule.csv"
    pd.DataFrame(
        [
            {"timestamp": "t0", "gross_exposure_scale": 0.5},
            {"timestamp": "t1", "gross_exposure_scale": 0.0},
        ]
    ).to_csv(schedule_path, index=False)

    params = replace(
        _tree_score_params(tmp_path),
        top_n=2,
        policy_entry_rank=2,
        policy_exit_rank=2,
        policy_gross_exposure_scale_path=schedule_path,
    )
    result = _build_target_weights(ranked, params)

    target_by_time = result.target_weights.groupby("signal_time")["target_weight"].sum()
    assert target_by_time["t0"] == pytest.approx(0.5)
    assert "t1" not in set(result.target_weights["signal_time"])
    assert result.diagnostics["gross_exposure_scaled_count"].sum() == 4


def test_tree_score_target_builder_applies_drawdown_brake_scale_cap(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
        ]
    )
    params = replace(
        _tree_score_params(tmp_path),
        top_n=2,
        policy_entry_rank=2,
        policy_exit_rank=2,
    )

    result = _build_target_weights(
        ranked,
        params,
        gross_exposure_scale_cap=0.4,
    )

    assert result.target_weights["target_weight"].sum() == pytest.approx(0.4)
    assert result.diagnostics.loc[0, "drawdown_brake_scale"] == pytest.approx(0.4)
    assert bool(result.diagnostics.loc[0, "drawdown_brake_active"]) is True


def test_tree_score_target_builder_uses_min_of_schedule_and_drawdown_brake(
    tmp_path,
) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
        ]
    )
    schedule_path = tmp_path / "gross_exposure_schedule.csv"
    pd.DataFrame([{"timestamp": "t0", "gross_exposure_scale": 0.7}]).to_csv(
        schedule_path,
        index=False,
    )
    params = replace(
        _tree_score_params(tmp_path),
        top_n=2,
        policy_entry_rank=2,
        policy_exit_rank=2,
        policy_gross_exposure_scale_path=schedule_path,
    )

    result = _build_target_weights(
        ranked,
        params,
        gross_exposure_scale_cap=0.4,
    )

    assert result.target_weights["target_weight"].sum() == pytest.approx(0.4)


def test_tree_score_target_builder_supports_cost_aware_optimizer(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.5, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.1, "rank": 2},
        ]
    )

    params = replace(
        _tree_score_params(tmp_path),
        trade_policy="cost_aware_optimizer",
        top_n=1,
        optimizer_candidate_rank=2,
        optimizer_score_to_edge_bps=100.0,
        optimizer_min_net_edge_bps=20.0,
    )
    result = _build_target_weights(ranked, params)

    assert result.target_weights["instrument_id"].tolist() == ["inst-a"]
    assert result.diagnostics.loc[0, "policy_id"] == "cost_aware_optimizer"


def test_tree_score_target_builder_enforces_remaining_path_turnover_budget(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.1, "rank": 2},
            {"signal_time": "t1", "instrument_id": "inst-b", "score": 0.9, "rank": 1},
            {"signal_time": "t1", "instrument_id": "inst-a", "score": 0.1, "rank": 2},
        ]
    )

    params = replace(
        _tree_score_params(tmp_path),
        trade_policy="cost_aware_optimizer",
        top_n=1,
        optimizer_candidate_rank=2,
        optimizer_score_to_edge_bps=100.0,
        policy_total_gross_turnover_budget=1.0,
    )
    result = _build_target_weights(ranked, params)

    diagnostics = result.diagnostics.set_index("timestamp")
    assert diagnostics.loc["t0", "dynamic_turnover_cap"] == pytest.approx(1.0)
    assert diagnostics.loc["t0", "planned_gross_turnover"] == pytest.approx(1.0)
    assert diagnostics.loc["t1", "dynamic_turnover_cap"] == pytest.approx(0.0)
    assert diagnostics.loc["t1", "planned_gross_turnover"] == pytest.approx(0.0)
    assert diagnostics.loc["t1", "turnover_path_budget_after"] == pytest.approx(0.0)
    assert result.diagnostics["planned_gross_turnover"].sum() == pytest.approx(1.0)


def test_tree_score_target_builder_optionally_paces_path_turnover_budget(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.1, "rank": 2},
            {"signal_time": "t1", "instrument_id": "inst-b", "score": 0.9, "rank": 1},
            {"signal_time": "t1", "instrument_id": "inst-a", "score": 0.1, "rank": 2},
        ]
    )

    params = replace(
        _tree_score_params(tmp_path),
        trade_policy="cost_aware_optimizer",
        top_n=1,
        optimizer_candidate_rank=2,
        optimizer_score_to_edge_bps=100.0,
        policy_total_gross_turnover_budget=1.0,
        policy_turnover_budget_pacing=1.0,
    )
    result = _build_target_weights(ranked, params)

    diagnostics = result.diagnostics.set_index("timestamp")
    assert diagnostics.loc["t0", "dynamic_turnover_cap"] == pytest.approx(0.5)
    assert diagnostics.loc["t0", "planned_gross_turnover"] == pytest.approx(0.5)
    assert diagnostics.loc["t1", "dynamic_turnover_cap"] == pytest.approx(0.5)
    assert diagnostics.loc["t1", "planned_gross_turnover"] == pytest.approx(0.5)


def test_tree_score_target_builder_replenishes_monthly_turnover_budget(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {
                "signal_time": "2024-01-02T10:00:00+08:00",
                "instrument_id": "inst-a",
                "score": 0.9,
                "rank": 1,
            },
            {
                "signal_time": "2024-01-02T10:00:00+08:00",
                "instrument_id": "inst-b",
                "score": 0.1,
                "rank": 2,
            },
            {
                "signal_time": "2024-01-02T10:00:00+08:00",
                "instrument_id": "inst-c",
                "score": 0.0,
                "rank": 3,
            },
            {
                "signal_time": "2024-01-03T10:00:00+08:00",
                "instrument_id": "inst-b",
                "score": 0.9,
                "rank": 1,
            },
            {
                "signal_time": "2024-01-03T10:00:00+08:00",
                "instrument_id": "inst-a",
                "score": 0.1,
                "rank": 2,
            },
            {
                "signal_time": "2024-01-03T10:00:00+08:00",
                "instrument_id": "inst-c",
                "score": 0.0,
                "rank": 3,
            },
            {
                "signal_time": "2024-02-01T10:00:00+08:00",
                "instrument_id": "inst-c",
                "score": 0.9,
                "rank": 1,
            },
            {
                "signal_time": "2024-02-01T10:00:00+08:00",
                "instrument_id": "inst-b",
                "score": 0.1,
                "rank": 2,
            },
            {
                "signal_time": "2024-02-01T10:00:00+08:00",
                "instrument_id": "inst-a",
                "score": 0.0,
                "rank": 3,
            },
        ]
    )

    params = replace(
        _tree_score_params(tmp_path),
        trade_policy="cost_aware_optimizer",
        top_n=1,
        optimizer_candidate_rank=3,
        optimizer_score_to_edge_bps=100.0,
        policy_total_gross_turnover_budget=1.0,
        policy_turnover_budget_period="month",
    )
    result = _build_target_weights(ranked, params)

    diagnostics = result.diagnostics.set_index("timestamp")
    assert diagnostics.loc["2024-01-02T10:00:00+08:00", "dynamic_turnover_cap"] == pytest.approx(1.0)
    assert diagnostics.loc["2024-01-02T10:00:00+08:00", "planned_gross_turnover"] == pytest.approx(1.0)
    assert diagnostics.loc["2024-01-03T10:00:00+08:00", "dynamic_turnover_cap"] == pytest.approx(0.0)
    assert diagnostics.loc["2024-01-03T10:00:00+08:00", "planned_gross_turnover"] == pytest.approx(0.0)
    assert diagnostics.loc["2024-02-01T10:00:00+08:00", "dynamic_turnover_cap"] == pytest.approx(1.0)
    assert diagnostics.loc["2024-02-01T10:00:00+08:00", "planned_gross_turnover"] == pytest.approx(1.0)
    assert diagnostics["turnover_budget_period_key"].tolist() == [
        "2024-01",
        "2024-01",
        "2024-02",
    ]


def test_tree_score_rank_limit_includes_policy_exit_rank(tmp_path) -> None:
    params = _tree_score_params(tmp_path)

    assert _score_rank_limit(params) == 2


def test_tree_score_rank_limit_includes_optimizer_candidate_rank(tmp_path) -> None:
    params = replace(
        _tree_score_params(tmp_path),
        trade_policy="cost_aware_optimizer",
        top_n=1,
        optimizer_candidate_rank=5,
    )

    assert _score_rank_limit(params) == 5


def test_tree_score_backtest_default_policy_cost_uses_round_trip_trading_costs(
    tmp_path,
) -> None:
    params = replace(
        _tree_score_params(tmp_path),
        commission_bps=3.0,
        slippage_bps=1.0,
        sell_stamp_tax_bps=5.0,
        policy_estimated_cost_bps=None,  # type: ignore[arg-type]
    )

    assert _resolved_policy_estimated_cost_bps(params) == pytest.approx(13.0)


def test_tree_score_loader_preserves_optimizer_forecast_columns(tmp_path) -> None:
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": f"inst-{index}",
                "score": 1.0 - index * 0.1,
                "expected_edge_bps": 100.0 - index,
                "risk_penalty_bps": float(index),
            }
            for index in range(3)
        ]
    ).to_parquet(score_path, index=False)
    params = replace(
        _tree_score_params(tmp_path),
        predictions_path=score_path,
        trade_policy="cost_aware_optimizer",
        top_n=1,
        optimizer_candidate_rank=3,
    )

    ranked = _load_ranked_score_signals(params, start="t0", end="t0")

    assert len(ranked) == 3
    assert "expected_edge_bps" in ranked.columns
    assert "risk_penalty_bps" in ranked.columns
    assert ranked.loc[0, "expected_edge_bps"] == pytest.approx(100.0)


def test_tree_score_sparse_executions_keep_only_targets_and_tracked_holdings() -> None:
    bars = pd.DataFrame(
        [
            _bar("t0", "inst-a"),
            _bar("t0", "inst-held"),
            _bar("t0", "inst-unused"),
            _bar("t1", "inst-a"),
            _bar("t1", "inst-held"),
            _bar("t1", "inst-unused"),
            _bar("t2", "inst-a"),
            _bar("t2", "inst-held"),
            _bar("t2", "inst-unused"),
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "signal_time": "t0",
                "instrument_id": "inst-a",
                "target_weight": 1.0,
            }
        ]
    )

    dense = _build_tree_score_executions(bars, signals)
    sparse = _build_tree_score_executions(
        bars,
        signals,
        tracked_instruments={"inst-held"},
        sparse=True,
    )

    assert set(dense["instrument_id"]) == {"inst-a", "inst-held", "inst-unused"}
    assert set(sparse["instrument_id"]) == {"inst-a", "inst-held"}
    assert len(sparse) == 6
    target = sparse.loc[
        (sparse["exec_time"] == "t1") & (sparse["instrument_id"] == "inst-a"),
        "target_weight",
    ]
    held = sparse.loc[
        (sparse["exec_time"] == "t1") & (sparse["instrument_id"] == "inst-held"),
        "target_weight",
    ]
    assert target.iloc[0] == pytest.approx(1.0)
    assert pd.isna(held.iloc[0])


def test_tree_score_segment_executions_match_sparse_filtered_window() -> None:
    bars = pd.DataFrame(
        [
            _bar("t0", "inst-a"),
            _bar("t0", "inst-held"),
            _bar("t0", "inst-unused"),
            _bar("t1", "inst-a"),
            _bar("t1", "inst-held"),
            _bar("t1", "inst-unused"),
            _bar("t2", "inst-a"),
            _bar("t2", "inst-held"),
            _bar("t2", "inst-unused"),
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "signal_time": "t0",
                "instrument_id": "inst-a",
                "target_weight": 1.0,
            }
        ]
    )

    sparse = _build_tree_score_executions(
        bars,
        signals,
        tracked_instruments={"inst-held"},
        sparse=True,
    )
    expected = sparse.loc[
        (sparse["exec_time"] > "t0") & (sparse["exec_time"] <= "t2")
    ].reset_index(drop=True)
    segment = _build_segment_tree_score_executions(
        bars,
        _bar_time_index(bars),
        signals,
        tracked_instruments={"inst-held"},
        start_exclusive="t0",
        end_inclusive="t2",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(segment, expected)


def _tree_score_params(tmp_path) -> TreeScoreBacktestParams:
    return TreeScoreBacktestParams(
        predictions_path=tmp_path,
        catalog_path=tmp_path / "catalog.duckdb",
        start="t0",
        end="t1",
        top_n=1,
        initial_cash=1_000_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        sell_stamp_tax_bps=0.0,
        min_commission=0.0,
        lot_size=100,
        trade_policy="rank_buffer_drop",
        rebalance_every_n_bars=1,
        hold_rank_buffer=None,
        policy_entry_rank=1,
        policy_exit_rank=2,
        policy_max_entries_per_rebalance=None,
        policy_max_exits_per_rebalance=None,
        policy_min_hold_bars=0,
        policy_min_expected_edge_bps=None,
        policy_estimated_cost_bps=0.0,
        policy_no_trade_weight_band=0.0,
        policy_partial_rebalance_rate=1.0,
        policy_max_gross_turnover_per_rebalance=None,
        policy_total_gross_turnover_budget=None,
        policy_turnover_budget_period="path",
        policy_turnover_budget_pacing=0.0,
        policy_gross_exposure_scale=1.0,
        policy_gross_exposure_scale_path=None,
        policy_drawdown_brake_threshold=None,
        policy_drawdown_brake_reduced_scale=0.5,
        optimizer_candidate_rank=None,
        optimizer_score_to_edge_bps=100.0,
        optimizer_min_net_edge_bps=0.0,
        optimizer_risk_penalty_multiplier=1.0,
        optimizer_weighting="utility",
        optimizer_max_name_weight=None,
        optimizer_max_gross_exposure_increase_per_rebalance=None,
        min_trade_weight=0.0,
        exclude_st=True,
        limit_up_bps=None,
        limit_down_bps=None,
        max_bar_turnover_participation=None,
        allow_same_bar_capacity=False,
        data_access_mode="data_portal",
        streaming_chunk="month",
        streaming_chunk_padding_days=0,
        output_dir=tmp_path,
    )


def _bar(timestamp: str, instrument_id: str) -> dict[str, object]:
    return {
        "bar_end_time": timestamp,
        "instrument_id": instrument_id,
        "canonical_code": instrument_id,
        "open_price": 10.0,
        "close_price": 10.0,
        "turnover": 1_000_000.0,
        "tradable_bar": True,
        "limit_up_open": False,
        "limit_down_open": False,
    }


def test_streaming_work_units_support_day_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    params = baseline_backtest.BacktestParams(
        catalog_path=Path("catalog.json"),
        start="2024-01-02T09:30:00+08:00",
        end="2024-01-03T15:00:00+08:00",
        top_n=1,
        initial_cash=1_000_000.0,
        lookback_bars=1,
        min_avg_turnover=None,
        liquidity_window_bars=1,
        commission_bps=0.0,
        slippage_bps=0.0,
        lot_size=100,
        max_symbols=None,
        output_dir=None,
        data_access_mode="fast_parquet",
        streaming_chunk="day",
        streaming_chunk_padding_days=0,
    )

    monkeypatch.setattr(
        baseline_backtest,
        "_minute_bar_files_for_range",
        lambda *_args, **_kwargs: [Path("bars.parquet")],
    )

    units = baseline_backtest._streaming_work_units(params)

    assert [unit.signal_start for unit in units] == [
        "2024-01-02T09:30:00+08:00",
        "2024-01-03T00:00:00+08:00",
    ]
    assert [unit.signal_end for unit in units] == [
        "2024-01-02T23:59:59+08:00",
        "2024-01-03T15:00:00+08:00",
    ]


def test_streaming_work_units_support_week_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    params = baseline_backtest.BacktestParams(
        catalog_path=Path("catalog.json"),
        start="2024-01-02T09:30:00+08:00",
        end="2024-01-12T15:00:00+08:00",
        top_n=1,
        initial_cash=1_000_000.0,
        lookback_bars=1,
        min_avg_turnover=None,
        liquidity_window_bars=1,
        commission_bps=0.0,
        slippage_bps=0.0,
        lot_size=100,
        max_symbols=None,
        output_dir=None,
        data_access_mode="fast_parquet",
        streaming_chunk="week",
        streaming_chunk_padding_days=0,
    )

    monkeypatch.setattr(
        baseline_backtest,
        "_minute_bar_files_for_range",
        lambda *_args, **_kwargs: [Path("bars.parquet")],
    )

    units = baseline_backtest._streaming_work_units(params)

    assert [unit.signal_start for unit in units] == [
        "2024-01-02T09:30:00+08:00",
        "2024-01-09T00:00:00+08:00",
    ]
    assert [unit.signal_end for unit in units] == [
        "2024-01-08T23:59:59+08:00",
        "2024-01-12T15:00:00+08:00",
    ]


def test_next_segment_end_prefers_next_signal() -> None:
    assert _next_segment_end(
        "2024-01-02T09:35:00+08:00",
        next_signal_time="2024-01-02T14:55:00+08:00",
        bar_times=["2024-01-02T09:35:00+08:00", "2024-01-02T09:40:00+08:00"],
    ) == "2024-01-02T14:55:00+08:00"


def test_streaming_with_drawdown_brake_dispatches_rebalance_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    params = _tree_score_params(tmp_path)
    params = replace(
        params,
        policy_drawdown_brake_threshold=-0.07,
        output_dir=tmp_path,
    )
    backtest_params = baseline_backtest.BacktestParams(
        catalog_path=Path("catalog.json"),
        start="2024-01-02T09:30:00+08:00",
        end="2024-01-03T15:00:00+08:00",
        top_n=1,
        initial_cash=1_000_000.0,
        lookback_bars=1,
        min_avg_turnover=None,
        liquidity_window_bars=1,
        commission_bps=0.0,
        slippage_bps=0.0,
        lot_size=100,
        max_symbols=None,
        output_dir=tmp_path,
        data_access_mode="fast_parquet",
        streaming_chunk="month",
        streaming_chunk_padding_days=0,
    )

    calls: list[str] = []

    monkeypatch.setattr(
        "examples.run_tree_score_backtest._run_tree_score_backtest_streaming_rebalance_drawdown",
        lambda *args, **kwargs: calls.append("rebalance") or {"summary": {}, "metrics": {}, "trades": pd.DataFrame(), "equity_curve": pd.DataFrame(), "final_positions": pd.DataFrame()},
    )

    result = _run_tree_score_backtest_streaming(params, backtest_params)

    assert calls == ["rebalance"]
    assert result["summary"] == {}
